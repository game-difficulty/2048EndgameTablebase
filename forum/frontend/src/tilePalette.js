import themes from "../../../docs_and_configs/themes.json";
import { resolveTileColors } from "../../../frontend/src/utils/tileColors.js";
import { readSharedTilePalette } from "../../../frontend/src/utils/sharedTilePalette.js";

// Use the same catalog, text contrast resolver and cross-subdomain cookie as Play.
const defaults = resolveTileColors(
  Array.from({ length: 36 }, (_, i) => themes.Default[i] || "#000000"),
);
const validColor = (value) =>
  typeof value === "string" &&
  /^(#[\da-f]{3,8}$|(?:rgb|hsl)a?\()/i.test(value.trim()) &&
  CSS.supports("color", value);

export function refreshTilePalette() {
  const shared = readSharedTilePalette();
  defaults.forEach((fallback, index) => {
    const entry = shared?.[index];
    const background = typeof entry === "string" ? entry : entry?.background;
    const valid = validColor(background);
    const color =
      valid && validColor(entry?.color)
        ? entry.color
        : valid && /^#[\da-f]{6}$/i.test(background)
          ? resolveTileColors([background])[0].color
          : fallback.color;
    const value = 2 ** (index + 1);
    document.documentElement.style.setProperty(
      `--color-tile-${value}`,
      valid ? background : fallback.background,
    );
    document.documentElement.style.setProperty(`--color-text-${value}`, color);
  });
}

export function tileStyle(value) {
  return value
    ? {
        background: `var(--color-tile-${value}, #000000)`,
        color: `var(--color-text-${value}, #f9f6f2)`,
      }
    : { background: "var(--color-empty)", color: "var(--text)" };
}
