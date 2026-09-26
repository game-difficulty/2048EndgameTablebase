export const SPEED_WINDOW_MS = 1000;
export const SPEED_SAMPLE_LIMIT = 100;
export const SPEED_REFRESH_MS = 100;

export function addSpeedSample(samples, time, limit = SPEED_SAMPLE_LIMIT) {
  const recent = samples.filter(sample => time - sample >= 0 && time - sample < SPEED_WINDOW_MS);
  recent.push(time);
  return recent.slice(-limit);
}

export function countSpeedSamples(samples, time) {
  return samples.reduce((count, sample) => {
    const age = time - sample;
    return count + Number(age >= 0 && age < SPEED_WINDOW_MS);
  }, 0);
}
