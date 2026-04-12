const parseRgbColor = (color) => {
  if (typeof color !== 'string') {
    return [160, 160, 160];
  }
  if (color.startsWith('#')) {
    const parts = color.slice(1).match(/.{2}/g);
    if (!parts) return [160, 160, 160];
    return parts.map((part) => parseInt(part, 16));
  }
  const rgbMatch = color.match(/\d+(\.\d+)?/g);
  if (!rgbMatch || rgbMatch.length < 3) {
    return [160, 160, 160];
  }
  return rgbMatch.slice(0, 3).map(Number);
};

const mixRgbColor = (color, target, ratio) => {
  const [r1, g1, b1] = parseRgbColor(color);
  const [r2, g2, b2] = target;
  const mix = (a, b) => Math.round(a + (b - a) * ratio);
  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
};

export const createResultBarGradient = (color) =>
  `linear-gradient(90deg, ${mixRgbColor(color, [255, 246, 224], 0.1)} 0%, ${color} 72%, ${mixRgbColor(color, [36, 28, 18], 0.05)} 100%)`;

