import i18n from '../../app/i18n.js';
import { serverErrorText } from './serverErrorText.js';

export function userError(error, fallback = '') {
  const locale = i18n.global.locale;
  return serverErrorText(error, locale?.value || locale, fallback);
}
