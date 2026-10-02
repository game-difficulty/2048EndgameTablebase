import i18n from '../app/i18n';

export function goalLabel(target) {
  const text = String(target);
  return text.startsWith('sum-') ? i18n.global.t('goals.sumLabel', { value: text.slice(4) }) : text;
}

export function validSumTarget(value) {
  const n = Number(value);
  return Number.isInteger(n) && n >= 4 && n < 16384 && n % 2 === 0;
}
