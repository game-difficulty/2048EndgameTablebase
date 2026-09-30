// Presentation only: never replace the error codes/messages used by controllers.
const messages = {
  'Invalid email or password.': '邮箱或密码不正确。',
  'Invalid email.': '邮箱地址格式不正确。',
  'Email domain is not currently supported for registration.': '暂不支持使用此邮箱域名注册。',
  'Email is already registered.': '该邮箱已注册，请登录或找回密码。',
  'Account is not active.': '该账号当前不可用，请联系站长。',
  'This account is managed by the site administrator.': '该账号由站长管理，如需注销请联系站长。',
  'Invalid current password.': '当前密码不正确。',
  'Invalid password.': '密码不正确。',
  'Invalid verification code.': '验证码不正确。',
  'Verification code not found.': '请先获取验证码。',
  'Verification code has expired.': '验证码已过期，请重新获取。',
  'Verification code has already been used.': '验证码已使用，请重新获取。',
  'Too many verification attempts.': '验证码尝试次数过多，请重新获取验证码。',
  'Too many verification emails. Please try again later.': '验证码邮件发送过于频繁，请稍后再试。',
  'Email service is not available.': '邮件服务暂时不可用，请稍后重试。',
  'Email service is not configured.': '邮件服务尚未配置，请联系站长。',
  'Invalid invite code.': '邀请码不正确。',
  'Invite code is disabled.': '该邀请码已停用。',
  'Invite code has expired.': '邀请码已过期。',
  'Invite code has already been used.': '邀请码已使用。',
  'Invite code is not valid for this email.': '该邀请码不适用于此邮箱。',
  'Invite code is not valid for this email domain.': '该邀请码不适用于此邮箱域名。',
  'Confirmation text is incorrect.': '确认文字不正确。',
  'Username is already in use.': '该用户名已被使用。',
  'Username must contain at least one letter or number.': '用户名至少需要包含一个文字或数字。',
  'Username may only contain letters, numbers, spaces, underscores, and hyphens.': '用户名只能包含文字、数字、空格、下划线和连字符。',
  'This username is reserved.': '该用户名为保留名称，请更换。',
  'User not found.': '用户不存在。',
  'Avatar not found.': '头像不存在。',
  'Avatar uploads are disabled for this account.': '此账号已停用头像上传。',
  'Authentication required.': '请先登录后再操作。',
  'A guest or user session is required.': '会话已失效，请刷新页面或重新登录。',
  'Invalid account status.': '账号状态无效。',
  'Token amount must be numeric.': '额度数量必须是数字。',
  'Token amount must be finite.': '请输入有效的额度数量。',
  'Token amount must not be negative.': '额度数量不能为负数。',
  'Token amount must be greater than zero.': '额度数量必须大于零。',
  'Token amount is too large.': '额度数量超出允许范围。',
  'Permanent balance must not be negative.': '常驻额度余额不能为负数。',
  'Invalid token adjustment mode.': '额度调整方式无效。',
  'Reason is required.': '请填写原因。',
  'Reason is too long.': '原因文字过长，请缩短。',
  'Payment amount must be numeric.': '付款金额必须是数字。',
  'Payment amount must not be negative.': '付款金额不能为负数。',
  'Leaderboard not found.': '该排行榜不存在。',
  'Analysis job not found.': '分析任务不存在或已过期。',
  'Minigame action failed.': '小游戏操作失败，请重试。',
};

const codes = {
  archive_duration_too_short: ['开始至结束的时长不能小于回放中已记录的步时总和。', 'The time from start to finish must be at least the sum of recorded move times.'],
  invalid_started_at: ['请填写有效的对局开始时间，且不能晚于结束时间。', 'Enter a valid game start time no later than the end time.'],
  AUTH_REQUIRED: ['请先登录后再操作。', 'Please sign in to continue.'],
  INSUFFICIENT_TOKENS: ['额度不足，请查看额度说明。', 'Insufficient tokens. See the quota guide.'],
  PROFILE_CHANGE_COOLDOWN: ['尚未到可修改时间，请稍后再试。', 'This profile field is still on cooldown.'],
  PROFILE_RATE_LIMIT: ['修改过于频繁，请稍后重试。', 'Too many profile changes. Please try again later.'],
  DISPLAY_NAME_TAKEN: ['该用户名已被使用。', 'This username is already in use.'],
  INVALID_AVATAR: ['头像文件无效，请选择有效的图片。', 'Invalid avatar. Please choose a valid image.'],
  AVATAR_UPLOAD_DISABLED: ['此账号已停用头像上传。', 'Avatar uploads are disabled for this account.'],
  invalid_preferences: ['设置内容无效，请检查后重试。', 'Invalid settings. Please check and retry.'],
  analysis_price_changed: ['分析费用已变化，请重新确认。', 'The analysis price changed. Please confirm again.'],
  analysis_catalog_changed: ['可用定式已变化，请重新选择。', 'The available tablebases changed. Please select again.'],
  analysis_source_missing: ['分析所需的回放已不存在。', 'The replay required for analysis is no longer available.'],
  invalid_analysis_job_id: ['分析任务编号无效。', 'Invalid analysis job ID.'],
  REMOTE_TABLEBASE_OFFLINE: ['所选定式暂不可用，请稍后重试。', 'The selected tablebase is temporarily unavailable.'],
  REMOTE_TABLEBASE_TIMEOUT: ['查表请求超时，请稍后重试。', 'The tablebase request timed out. Please try again.'],
  TABLEBASE_BUSY: ['查表服务繁忙，正在等待重试。', 'The tablebase service is busy. Waiting to retry.'],
};

export function serverErrorText(error, locale = 'en', fallback = '') {
  const zh = /^zh(?:-|$)/i.test(String(locale));
  const detail = error?.detail;
  const code = detail?.code || error?.code || (typeof detail === 'string' ? detail : '');
  const raw = typeof error === 'string' ? error : (typeof detail === 'string' ? detail : detail?.message) || error?.message || '';
  const message = String(raw).trim();
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key) ? object[key] : undefined;
  const pair = own(codes, code) || own(codes, message);
  if (pair) return pair[zh ? 0 : 1];
  if (message && Object.prototype.hasOwnProperty.call(messages, message)) return zh ? messages[message] : message;
  if (code === 'EMAIL_CODE_COOLDOWN') {
    const seconds = Math.max(1, Math.ceil(Number(detail?.retry_after_seconds) || 300));
    return zh ? `请等待 ${seconds} 秒后再获取验证码。` : `Please wait ${seconds} seconds before requesting another code.`;
  }
  const length = /^(Password|Username) must contain at (least|most) (\d+) characters\.$/.exec(message);
  if (length) return zh ? `${length[1] === 'Password' ? '密码' : '用户名'}${length[2] === 'least' ? '至少' : '最多'}需要 ${length[3]} 个字符。` : message;
  if (error?.name === 'AbortError' || /timeout|timed out/i.test(message)) {
    return zh ? '请求超时，请稍后重试。' : 'The request timed out. Please try again.';
  }
  if (/failed to fetch|network(?:error| request failed)|load failed/i.test(message)) {
    return zh ? '网络连接失败，请检查网络后重试。' : 'Could not connect. Please check your network and retry.';
  }
  const status = Number(error?.status || /(?:HTTP|failed:)\s*(\d{3})\b/i.exec(message)?.[1]);
  const statusMessages = {
    401: ['登录已失效，请重新登录。', 'Your session expired. Please sign in again.'],
    402: codes.INSUFFICIENT_TOKENS,
    403: ['当前账号没有此操作权限。', 'You do not have permission to perform this action.'],
    404: ['请求的内容不存在或已过期。', 'The requested content is missing or has expired.'],
    409: ['当前状态已变化，请刷新后重试。', 'The current state changed. Please refresh and retry.'],
    413: ['提交的文件或内容过大。', 'The submitted file or content is too large.'],
    422: ['提交的信息格式不正确，请检查后重试。', 'Invalid input. Please check and retry.'],
    429: ['操作过于频繁，请稍后重试。', 'Too many requests. Please try again later.'],
  };
  if (statusMessages[status]) return statusMessages[status][zh ? 0 : 1];
  if (status >= 500) return zh ? '服务器暂时不可用，请稍后重试。' : 'The server is temporarily unavailable. Please try again later.';
  // Keep already-localized messages; unknown diagnostics stay on the original error.
  if (zh && /[\u3400-\u9fff]/.test(message)) return message;
  if (!zh && message && !/^[a-z][a-z0-9_]+$/.test(message) && !/^</.test(message)) return message;
  return fallback || (zh ? '操作暂时未能完成，请稍后重试。' : 'The operation could not be completed. Please try again.');
}
