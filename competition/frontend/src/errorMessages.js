// Competition API and room socket errors share stable codes. Keep user-facing
// wording here instead of displaying the backend's English diagnostic text.
export const ERROR_MESSAGES = Object.freeze({
  REMATCH_WINDOW_CLOSED: '已进入布阵阶段，不再因落位错误重赛。',
  SCHEDULE_SEATS_FIXED: '请按报名表上的队内序号落座。',
  AUTH_REQUIRED: '请先登录后再操作。',
  ACTIVE_PLAYER_REQUIRED: '只有本场项目的出战选手可以执行此操作。',
  BLIND_ALREADY_SUBMITTED: '本队已提交盲选结果。',
  CAPTAIN_REQUIRED: '只有队长席位的选手可以执行此操作。',
  CLOCK_NOT_INITIALIZED: '队伍计时器尚未就绪，请联系裁判。',
  COMMAND_ID_REUSED: '这次操作已提交，请刷新房间状态。',
  DRAFT_NOT_INITIALIZED: '选 Ban 流程尚未开始。',
  DUPLICATE_PROJECT: '项目池中不能有重复的项目名称或标识。',
  GAME_NOT_COMPLETE: '双方都完成本场项目后才能继续。',
  INCOMPLETE_TEAM: '双方队伍需保持满员。',
  INVALID_CLIENT_STATE: '上传的对局状态无效，请刷新后重试。',
  INVALID_COMMAND_ID: '操作标识无效，请重试。',
  INVALID_DRAFT_PHASE: '当前不在可选 Ban 阶段。',
  INVALID_GAME_PHASE: '当前对局阶段不能执行此操作。',
  INVALID_ISSUE_CATEGORY: '请选择有效的问题类别。',
  INVALID_ISSUE_STATUS: '请选择有效的问题处理状态。',
  INVALID_LINEUP: '请为每局安排选手，并满足本房间的出场限制。',
  INVALID_ROOM_RULES: '请检查 BP 步骤、每队人数、计时及项目池是否符合房间规则。',
  INVALID_DRAFT_SELECTION: '选禁数量不符，或项目已被选择、禁用。',
  NOT_YOUR_TURN: '当前不是本方的选禁回合。',
  INVALID_LINEUP_PHASE: '当前不在秘密布阵阶段。',
  INVALID_MOVE: '这一步操作无效，请重试。',
  INVALID_NAME: '比赛名称须为 2–100 个字符。',
  INVALID_POSITION: '请选择本房间范围内的有效席位。',
  INVALID_PRACTICE_RESULT: '本次试玩结果无法记录。',
  INVALID_PROJECT: '项目名称须为 1–80 个字符。',
  INVALID_PROJECT_KEY: '项目标识只能使用小写字母、数字、点、短横线或下划线。',
  INVALID_PROJECT_POOL: '项目池须包含 1–32 个项目，且满足所选流程的最低要求。',
  INVALID_READINESS_ROLE: '准备身份无效，请刷新后重试。',
  INVALID_REASON: '原因说明须为 3–500 个字符。',
  INVALID_SCORE: '得分不能为负数。',
  INVALID_SIDE: '队伍选择无效。',
  INVALID_STAFF_ROLE: '工作人员身份无效。',
  INVALID_SUSPENSION_REASON: '请选择有效的暂停原因。',
  INVALID_WINNER: '请选择黄方、白方或平局。',
  ISSUE_ALREADY_CLOSED: '该问题已处理完毕。',
  ISSUE_NOT_FOUND: '找不到该问题报告。',
  LINEUP_ALREADY_SUBMITTED: '本队已提交布阵。',
  LINEUP_INCOMPLETE: '出战阵容尚未完整，请联系裁判。',
  LINEUP_NOT_INITIALIZED: '布阵阶段尚未就绪。',
  LIVE_INTERNAL_AUTH_REQUIRED: '直播服务未通过身份验证。',
  LIVE_ROOM_NOT_FOUND: '找不到对应的直播房间。',
  MATCH_ALREADY_SUSPENDED: '比赛已经暂停。',
  MATCH_NOT_ACTIVE: '当前比赛未处于可操作阶段。',
  MATCH_NOT_INITIALIZED: '比赛尚未就绪，请稍后重试。',
  MATCH_NOT_SUSPENDED: '比赛当前没有暂停。',
  MATCH_SUSPENDED: '比赛已由裁判暂停。',
  NOT_ACTIVE_SIDE: '当前轮到对方队长操作。',
  NOT_SEATED: '你尚未在该房间落座。',
  ORGANIZER_REQUIRED: '只有主办方可以执行此操作。',
  PICK_BAN_CONFLICT: '选择和禁用的项目不能相同。',
  PLAYER_REQUIRED: '只有已落座的选手可以提交比赛问题。',
  PROJECT_ADAPTER_UNAVAILABLE: '该项目暂不可用，请联系主办方。',
  PROJECT_ALREADY_COMPLETE: '你的本场项目已完成。',
  PROJECT_NOT_FOUND: '项目池中找不到所选项目。',
  PROJECT_POOL_EXHAUSTED: '项目池中已没有足够的可选项目。',
  PROJECT_UNAVAILABLE: '所选项目当前不可用，请重新选择。',
  READINESS_INCOMPLETE: '请等待双方出战选手和队长全部准备。',
  READY_CHECK_CLOSED: '当前不能修改准备状态。',
  REFEREE_REQUIRED: '只有主办方或裁判可以执行此操作。',
  RESULT_ALREADY_CONFIRMED: '本队已确认该场结果。',
  RESUME_READINESS_INCOMPLETE: '请等待双方队长确认恢复比赛。',
  ROOM_CLOSE_UNAVAILABLE: '抽签开始后不能关闭房间。',
  ROOM_CODE_EXHAUSTED: '暂时无法分配房间码，请稍后重试。',
  ROOM_CODE_TAKEN: '该房间码已被使用。',
  ROOM_NOT_FOUND: '找不到该比赛房间，请检查房间码。',
  REMOVED_FROM_ROOM: '你已被移出此房间，请联系房主或赛事管理员。',
  ROOM_MANAGER_REQUIRED: '仅赛事管理员或该房间的房主可以管理人员。',
  CANNOT_REMOVE_SELF: '不能移出自己。',
  MEMBER_REMOVAL_HOLD: '参赛人员已被移出，需房主或赛事管理员处理后继续。',
  SURRENDER_NOT_ALLOWED: '仅当对方本局已完赛且未认输时，才可认输。',
  RESULT_REST_REQUIRED: '本局休整尚未结束，请等待 30 秒展示完毕。',
  SEAT_TAKEN: '该席位已有人落座。',
  SEATING_CLOSED: '该房间已停止选座。',
  SEATS_INCOMPLETE: '本队全部选手均需落座。',
  SEATS_LOCKED: '队长准备后，席位已锁定。',
  STALE_PHASE: '比赛阶段已变化，请刷新房间后重试。',
  STALE_RESULT: '比赛结果已变化，请重新确认。',
  STATE_TOO_LARGE: '对局状态过大，无法同步。',
  TEAM_CLOCK_EXPIRED: '本队包干时间已用尽，正在结算比赛。',
  UNKNOWN_PRACTICE_PROJECT: '找不到该试玩项目。',
  USER_NOT_FOUND: '找不到该用户或用户已停用。',
});

export function userFacingError(cause) {
  const code = cause?.code;
  if (code && ERROR_MESSAGES[code]) return ERROR_MESSAGES[code];
  const message = typeof cause === 'string' ? cause : cause?.message;
  if (typeof message === 'string' && /[\u3400-\u9fff]/u.test(message)) return message;
  if (cause?.status === 401) return ERROR_MESSAGES.AUTH_REQUIRED;
  if (cause?.status === 403) return '没有执行此操作的权限。';
  if (cause?.status === 404) return '找不到对应内容，请检查后重试。';
  if (cause?.status === 409) return '房间状态已变化，请刷新后重试。';
  if (cause?.status === 422) return '提交的内容无效，请检查后重试。';
  if (cause?.status >= 500) return '服务暂时不可用，请稍后重试。';
  if (cause instanceof TypeError || /(?:failed to fetch|networkerror)/i.test(message || '')) {
    return '网络连接失败，请检查网络后重试。';
  }
  return '操作失败，请稍后重试。';
}
