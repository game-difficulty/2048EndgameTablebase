import fs from "node:fs";
import { encodeVerseHistory } from "./verse-history-codec.mjs";

const [, , inputPath, outputPath] = process.argv;
if (!inputPath || !outputPath) {
  process.stderr.write("Usage: node convert-json-to-vhr.mjs <input.json> <output.vhr>\n");
  process.exit(2);
}

const input = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const encoded = encodeVerseHistory(input);
fs.writeFileSync(outputPath, encoded);
process.stdout.write(`${outputPath}: ${encoded.length} bytes\n`);
