import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { serverErrorText } from '../src/services/errors/serverErrorText.js';

test('login errors translate without exposing which credential was wrong', () => {
  const error = Object.assign(new Error('Invalid email or password.'), { status: 401 });
  assert.equal(serverErrorText(error, 'zh'), '邮箱或密码不正确。');
  assert.equal(serverErrorText(error, 'en'), 'Invalid email or password.');
  assert.equal(error.message, 'Invalid email or password.');
  assert.equal(error.status, 401);
  assert.equal(serverErrorText({ detail: error.message, status: 401 }, 'zh-CN'), '邮箱或密码不正确。');
});

test('all literal auth service validation errors have specific Chinese messages', () => {
  const source = readFileSync(new URL('../../backend/auth/service.py', import.meta.url), 'utf8');
  const messages = new Set([...source.matchAll(/raise (?:ValueError|RuntimeError|FileNotFoundError)\("([^"\n]+)"\)/g)].map(match => match[1]));
  assert.ok(messages.size > 20);
  for (const message of messages) {
    const result = serverErrorText(new Error(message), 'zh', 'MISSING');
    assert.notEqual(result, 'MISSING', message);
    assert.match(result, /[\u3400-\u9fff]/, message);
    assert.equal(serverErrorText(new Error(message), 'en'), message);
  }
});

test('structured errors, validation arrays and HTTP failures are readable', () => {
  assert.equal(serverErrorText({ detail: { code: 'INSUFFICIENT_TOKENS', message: 'low' }, status: 402 }, 'zh'), '额度不足，请查看额度说明。');
  assert.equal(serverErrorText({ detail: { code: 'EMAIL_CODE_COOLDOWN', retry_after_seconds: 17.2 } }, 'zh'), '请等待 18 秒后再获取验证码。');
  assert.equal(serverErrorText({ detail: [{ msg: 'Field required' }], status: 422 }, 'zh'), '提交的信息格式不正确，请检查后重试。');
  assert.match(serverErrorText('HTTP 503', 'zh'), /服务器/);
  assert.match(serverErrorText('Request failed: 429', 'zh'), /频繁/);
  assert.match(serverErrorText(new TypeError('Failed to fetch'), 'zh'), /网络/);
  assert.match(serverErrorText({ name: 'AbortError' }, 'zh'), /超时/);
});

test('locale is evaluated per call and fallbacks never alter controller errors', () => {
  const error = Object.freeze({ code: 'new_server_code', message: 'new_server_code' });
  assert.equal(serverErrorText(error, 'zh', '无法完成此操作。'), '无法完成此操作。');
  assert.equal(serverErrorText(error, 'en', 'Cannot complete this action.'), 'Cannot complete this action.');
  assert.equal(serverErrorText('用户名需要修改。', 'zh'), '用户名需要修改。');
  assert.equal(serverErrorText('New English diagnostic.', 'en'), 'New English diagnostic.');
  assert.doesNotMatch(serverErrorText('<html>502 Bad Gateway</html>', 'zh'), /html|Gateway/);
  assert.equal(typeof serverErrorText({ code: 'constructor' }, 'zh'), 'string');
  assert.equal(error.code, 'new_server_code');
});

test('shared auth and live login use the formatter at the display boundary', () => {
  for (const name of ['AuthPage', 'AccountSecurityDialog']) {
    const source = readFileSync(new URL(`../src/features/auth/${name}.vue`, import.meta.url), 'utf8');
    assert.match(source, /showMessage\(userError\(error\), 'error'\)/);
    assert.doesNotMatch(source, /showMessage\(error\.message/);
  }
  const live = readFileSync(new URL('../src/live/LivePage.vue', import.meta.url), 'utf8');
  assert.match(live, /loginError\.value = serverErrorText\(error, lang\.value\)/);
  const api = readFileSync(new URL('../src/live/roomContext.js', import.meta.url), 'utf8');
  assert.match(api, /detail: payload.detail/);
});

test('standalone replay request errors are localized without changing decoder errors', () => {
  const source = readFileSync(new URL('../public/verse-replay/i18n.js', import.meta.url), 'utf8');
  for (const lang of ['zh', 'en']) {
    const window = { location: { search: `?lang=${lang}` }, navigator: { language: lang } };
    runInNewContext(source, { window, URLSearchParams });
    const { requestError, translate } = window.ReplayI18n;
    assert.match(requestError({ status: 401 }), lang === 'zh' ? /登录/ : /sign in/);
    assert.match(requestError(new Error('HTTP 404')), lang === 'zh' ? /过期/ : /expired/);
    assert.match(requestError(new Error('Failed to fetch')), lang === 'zh' ? /网络/ : /Network/);
    assert.equal(requestError(new Error('回放格式错误')), translate('回放格式错误'));
    assert.equal(/[\u3400-\u9fff]/.test(translate('无法载入对局回放：' + requestError({ status: 403 }))), lang === 'zh');
  }
});
