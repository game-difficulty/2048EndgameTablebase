async page => {
  const CONFIG = "__CONFIG__";
  const VARIANTS = ["4x4", "3x4", "3x3", "2x4"];
  const VARIANT_IDS = { "4x4": 0, "3x4": 1, "3x3": 2, "2x4": 3 };
  const MONTHS = {
    January: "01", February: "02", March: "03", April: "04",
    May: "05", June: "06", July: "07", August: "08",
    September: "09", October: "10", November: "11", December: "12",
  };

  function parseDisplayedTime(display) {
    const match = display.match(
      /^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})\s+at\s+(\d{2}):(\d{2}):(\d{2})$/,
    );
    if (!match) return null;
    const [, month, day, year, hour, minute, second] = match;
    return Math.floor(Date.UTC(
      Number(year), Number(MONTHS[month]) - 1, Number(day),
      Number(hour), Number(minute), Number(second),
    ) / 1000);
  }

  function decodeBoardExponents(code, columns, rows) {
    if (code.length !== columns * rows) {
      throw new Error(`Board code length ${code.length} does not match ${columns}x${rows}`);
    }
    return [...code].map((character) => {
      const exponent = Number.parseInt(character, 36);
      if (!Number.isInteger(exponent) || exponent < 0 || exponent > 31) {
        throw new Error(`Unknown board character: ${character}`);
      }
      return exponent;
    });
  }

  function writeVarUint(output, value) {
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`Invalid unsigned integer: ${value}`);
    do {
      let byte = value % 128;
      value = Math.floor(value / 128);
      if (value > 0) byte += 128;
      output.push(byte);
    } while (value > 0);
  }

  function writeUint32(output, value) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
      throw new Error(`Value is outside uint32: ${value}`);
    }
    output.push(value & 255, Math.floor(value / 256) & 255, Math.floor(value / 65536) & 255, Math.floor(value / 16777216) & 255);
  }

  function packBoard(output, exponents) {
    let buffer = 0;
    let bitCount = 0;
    for (const exponent of exponents) {
      buffer += exponent * (2 ** bitCount);
      bitCount += 5;
      while (bitCount >= 8) {
        output.push(buffer & 255);
        buffer = Math.floor(buffer / 256);
        bitCount -= 8;
      }
    }
    if (bitCount > 0) output.push(buffer & 255);
  }

  function encodeSegment(variant, records, stats, collectedAtSeconds) {
    const output = [0x56, 0x48, 0x53, 0x31]; // VHS1
    output.push(VARIANT_IDS[variant]);
    writeUint32(output, collectedAtSeconds);
    writeVarUint(output, stats.pageDeclared);
    writeVarUint(output, stats.rawRead);
    writeVarUint(output, records.length);
    writeVarUint(output, stats.maximumScore);

    const ordered = [...records].sort((left, right) =>
      left.playedAtSeconds - right.playedAtSeconds || right.score - left.score,
    );
    let previousTime = 0;
    for (let index = 0; index < ordered.length; index += 1) {
      const record = ordered[index];
      if (index === 0) writeUint32(output, record.playedAtSeconds);
      else writeVarUint(output, record.playedAtSeconds - previousTime);
      writeVarUint(output, record.score);
      packBoard(output, record.tileExponents);
      previousTime = record.playedAtSeconds;
    }
    return new Uint8Array(output);
  }

  async function saveBinary(binary, outputPath, filename) {
    const downloadPromise = page.waitForEvent("download");
    await page.evaluate(({ contents, downloadName }) => {
      const blob = new Blob([new Uint8Array(contents)], { type: "application/octet-stream" });
      const anchor = document.createElement("a");
      anchor.href = URL.createObjectURL(blob);
      anchor.download = downloadName;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    }, { contents: [...binary], downloadName: filename });
    const download = await downloadPromise;
    await download.saveAs(outputPath);
  }

  async function selectIndexFor(values) {
    return page.locator("select").evaluateAll((selects, expectedValues) => {
      return selects.findIndex((select) => {
        const actual = [...select.options].map((option) => option.value);
        return expectedValues.every((value) => actual.includes(value));
      });
    }, values);
  }

  async function rowFingerprint() {
    return page.locator(".user-score-display-header").evaluateAll((headers) =>
      headers.slice(0, 3).map((header) => header.childNodes[1]?.textContent?.trim() || header.textContent.trim()).join("|"),
    );
  }

  async function readPageStats() {
    await page.waitForFunction(() => {
      const text = document.body.innerText;
      const maximum = text.match(/(?:\u6700\u9ad8\u5206|Highest Score)\s*[\uff1a:]\s*([\d,]+)/i);
      const total = text.match(/(?:\u603b\u5c40\u6570|Total Games)\s*[\uff1a:]\s*([\d,]+)/i);
      const firstHeading = document.querySelector(".user-score-display-header")?.textContent?.trim();
      const firstScore = firstHeading?.match(/^([\d,]+)\s+-/);
      if (!maximum || !total || !firstScore) return false;
      return Number(maximum[1].replaceAll(",", "")) === Number(firstScore[1].replaceAll(",", ""));
    }, undefined, { timeout: 30000 });
    const text = await page.locator("body").innerText();
    const maximum = text.match(/(?:\u6700\u9ad8\u5206|Highest Score)\s*[\uff1a:]\s*([\d,]+)/i);
    const total = text.match(/(?:\u603b\u5c40\u6570|Total Games)\s*[\uff1a:]\s*([\d,]+)/i);
    const rating = text.match(/(?:\u7b49\u7ea7\u5206|Rating)\s*[\uff1a:]\s*([\d,]+)/i);
    if (!maximum || !total) throw new Error("Cannot read the variant summary from the page");
    return {
      maximumScore: Number(maximum[1].replaceAll(",", "")),
      totalGames: Number(total[1].replaceAll(",", "")),
      rating: rating ? Number(rating[1].replaceAll(",", "")) : null,
    };
  }

  async function loadEveryVisibleBatch(expectedTotal) {
    let previousCount = -1;
    let stalledRounds = 0;
    while (true) {
      const beforeScroll = await page.locator(".user-score-display").count();
      if (beforeScroll >= expectedTotal) {
        if (beforeScroll !== expectedTotal) {
          throw new Error(`Page rendered ${beforeScroll} rows but declares ${expectedTotal}`);
        }
        return beforeScroll;
      }
      await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
      await page.waitForTimeout(1500);
      const count = await page.locator(".user-score-display").count();
      if (count === previousCount) stalledRounds += 1;
      else stalledRounds = 0;
      previousCount = count;
      if (stalledRounds >= 20) {
        throw new Error(`Infinite scrolling stalled at ${count} of ${expectedTotal} rows`);
      }
    }
  }

  async function collectCurrentVariant(variant) {
    const [columns, rows] = variant.split("x").map(Number);
    const pageStats = await readPageStats();
    const total = await loadEveryVisibleBatch(pageStats.totalGames);
    const rowLocator = page.locator(".user-score-display");
    const records = [];

    for (let start = 0; start < total; start += CONFIG.batchSize) {
      const end = Math.min(start + CONFIG.batchSize, total);
      await rowLocator.evaluateAll((elements, range) => {
        for (let index = range.start; index < range.end; index += 1) {
          const row = elements[index];
          if (!row.querySelector('img[src*="/board/"]')) row.querySelector("button.board-toggle")?.click();
        }
      }, { start, end });
      await page.waitForFunction(({ startIndex, endIndex }) => {
        const elements = [...document.querySelectorAll(".user-score-display")];
        return elements.slice(startIndex, endIndex).every((row) => row.querySelector('img[src*="/board/"]'));
      }, { startIndex: start, endIndex: end }, { timeout: 15000 });

      const batch = await rowLocator.evaluateAll((elements, range) => {
        return elements.slice(range.start, range.end).map((row) => {
          const header = row.querySelector(".user-score-display-header");
          const textNode = [...header.childNodes].find((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
          return {
            headerText: (textNode?.textContent || header.textContent).trim(),
            source: row.querySelector('img[src*="/board/"]')?.src || null,
          };
        });
      }, { start, end });

      for (let offset = 0; offset < batch.length; offset += 1) {
        const index = start + offset;
        const { headerText, source } = batch[offset];
        const match = headerText.match(/^([\d,]+)\s+-\s+(.+)$/);
        if (!match) throw new Error(`Cannot parse game heading: ${headerText}`);
        if (!source) throw new Error(`Game ${index + 1} has no board image URL`);
        const boardPath = source.replace(/^https?:\/\/[^/]+/i, "").split("?", 1)[0];
        const boardMatch = boardPath.match(/^\/board\/([^/]+)\/([0-9a-z]+)$/i);
        if (!boardMatch || boardMatch[1] !== variant) {
          throw new Error(`Game ${index + 1} board URL does not match ${variant}: ${boardPath}`);
        }
        const playedAtDisplay = match[2].trim();
        const playedAtSeconds = parseDisplayedTime(playedAtDisplay);
        if (playedAtSeconds === null) throw new Error(`Cannot parse game time: ${playedAtDisplay}`);
        records.push({
          score: Number(match[1].replaceAll(",", "")),
          playedAtSeconds,
          tileExponents: decodeBoardExponents(boardMatch[2].toLowerCase(), columns, rows),
        });
      }

      await rowLocator.evaluateAll((elements, range) => {
        for (let index = range.start; index < range.end; index += 1) {
          const row = elements[index];
          if (row.querySelector('img[src*="/board/"]')) row.querySelector("button.board-toggle")?.click();
        }
      }, { start, end });
      await page.waitForTimeout(CONFIG.batchDelayMs);
    }

    const unique = new Map();
    for (const record of records) {
      const key = `${record.score}|${record.playedAtSeconds}|${record.tileExponents.join(",")}`;
      if (!unique.has(key)) unique.set(key, record);
    }
    const actualMaximum = records.reduce((maximum, record) => Math.max(maximum, record.score), 0);
    if (records.length !== pageStats.totalGames) {
      throw new Error(`${variant} collected ${records.length} records but the page declares ${pageStats.totalGames}`);
    }
    if (actualMaximum !== pageStats.maximumScore) {
      throw new Error(`${variant} maximum score ${actualMaximum} does not match page value ${pageStats.maximumScore}`);
    }
    return {
      records: [...unique.values()],
      pageStats,
      rawRead: records.length,
      duplicateCount: records.length - unique.size,
    };
  }

  await page.waitForLoadState("domcontentloaded");
  const variantSelectIndex = await selectIndexFor(VARIANTS);
  const sortSelectIndex = await selectIndexFor(["score", "date"]);
  const directionSelectIndex = await selectIndexFor(["desc", "asc"]);
  if (variantSelectIndex < 0 || sortSelectIndex < 0 || directionSelectIndex < 0) {
    throw new Error("History filter controls were not found; the page structure may have changed");
  }

  const variantSelect = page.locator("select").nth(variantSelectIndex);
  await page.locator("select").nth(sortSelectIndex).selectOption("score");
  await page.locator("select").nth(directionSelectIndex).selectOption("desc");
  await page.locator(".user-score-display").first().waitFor({ state: "attached", timeout: 30000 });
  await page.waitForTimeout(CONFIG.initialSettleMs);

  const summaries = {};

  for (const variant of CONFIG.variantsToCollect) {
    const selectedBefore = await variantSelect.inputValue();
    const fingerprintBefore = await rowFingerprint();
    if (selectedBefore !== variant) {
      await variantSelect.selectOption(variant);
      await page.waitForFunction(
        ({ selectIndex, expectedVariant, oldFingerprint }) => {
          const select = document.querySelectorAll("select")[selectIndex];
          const headers = [...document.querySelectorAll(".user-score-display-header")];
          const fingerprint = headers.slice(0, 3).map((header) => {
            const textNode = [...header.childNodes].find((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
            return (textNode?.textContent || header.textContent).trim();
          }).join("|");
          return select?.value === expectedVariant && headers.length > 0 && fingerprint !== oldFingerprint;
        },
        { selectIndex: variantSelectIndex, expectedVariant: variant, oldFingerprint: fingerprintBefore },
        { timeout: 30000 },
      );
      await page.waitForTimeout(CONFIG.variantSettleMs);
    }
    const collected = await collectCurrentVariant(variant);
    const binary = encodeSegment(variant, collected.records, {
      pageDeclared: collected.pageStats.totalGames,
      rawRead: collected.rawRead,
      maximumScore: collected.pageStats.maximumScore,
    }, Math.floor(Date.now() / 1000));
    await saveBinary(binary, CONFIG.outputPaths[variant], `${variant}.vhs`);
    summaries[variant] = {
      pageTotal: collected.pageStats.totalGames,
      rawRead: collected.rawRead,
      uniqueCount: collected.records.length,
      pageMaximumScore: collected.pageStats.maximumScore,
      pageRating: collected.pageStats.rating,
      duplicates: collected.duplicateCount,
      bytes: binary.length,
      outputPath: CONFIG.outputPaths[variant],
    };
  }

  return {
    summaries,
  };
}
