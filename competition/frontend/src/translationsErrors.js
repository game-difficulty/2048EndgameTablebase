const entries=`
已进入布阵阶段，不再因落位错误重赛。|Lineup selection has begun. Incorrect seating can no longer trigger a rematch.
请按报名表上的队内序号落座。|Take the seat matching your registered team position.
请先登录后再操作。|Please sign in first.
只有本场项目的出战选手可以执行此操作。|Only the player assigned to this game can do this.
本队已提交盲选结果。|Your team has already submitted its blind pick.
只有队长席位的选手可以执行此操作。|Only the captain can do this.
队伍计时器尚未就绪，请联系裁判。|Team clocks are not initialized. Contact a referee.
这次操作已提交，请刷新房间状态。|This action was already submitted. Refresh the room.
选 Ban 流程尚未开始。|Pick/ban has not started.
项目池中不能有重复的项目名称或标识。|Game names and identifiers must be unique in the pool.
双方都完成本场项目后才能继续。|Both sides must finish this game before continuing.
双方队伍需保持满员。|Both teams must have all players seated.
上传的对局状态无效，请刷新后重试。|Invalid game state. Refresh and try again.
操作标识无效，请重试。|Invalid action ID. Try again.
当前不在可选 Ban 阶段。|Pick/ban is not available in this phase.
当前对局阶段不能执行此操作。|This action is unavailable in the current phase.
请选择有效的问题类别。|Select a valid issue category.
请选择有效的问题处理状态。|Select a valid issue status.
A、B、C 三场必须分别安排一名不同的队员。|Assign a different player to each of games A, B, and C.
当前不在秘密布阵阶段。|Secret lineup selection is not active.
这一步操作无效，请重试。|Invalid move. Try again.
比赛名称须为 2–100 个字符。|Match names must contain 2–100 characters.
请选择 1、2 或 3 号席位。|Choose seat 1, 2, or 3.
本次试玩结果无法记录。|This practice result could not be saved.
项目名称须为 1–80 个字符。|Game names must contain 1–80 characters.
项目标识只能使用小写字母、数字、点、短横线或下划线。|Game IDs may contain lowercase letters, digits, dots, hyphens, and underscores only.
项目池须包含 5–32 个项目。|The pool must contain 5–32 games.
准备身份无效，请刷新后重试。|Invalid readiness role. Refresh and retry.
原因说明须为 3–500 个字符。|Reasons must contain 3–500 characters.
得分不能为负数。|Scores cannot be negative.
队伍选择无效。|Invalid team selection.
工作人员身份无效。|Invalid staff role.
请选择有效的暂停原因。|Select a valid pause reason.
请选择黄方、白方或平局。|Choose Yellow, White, or Draw.
该问题已处理完毕。|This issue is already closed.
找不到该问题报告。|Issue report not found.
本队已提交布阵。|Your team has already submitted its lineup.
出战阵容尚未完整，请联系裁判。|The lineup is incomplete. Contact a referee.
布阵阶段尚未就绪。|Lineup selection is not initialized.
直播服务未通过身份验证。|Live service authentication failed.
找不到对应的直播房间。|Live room not found.
比赛已经暂停。|The match is already paused.
当前比赛未处于可操作阶段。|This action is unavailable in the current match phase.
比赛尚未就绪，请稍后重试。|The match is not ready. Try again shortly.
比赛当前没有暂停。|The match is not paused.
比赛已由裁判暂停。|The referee has paused the match.
当前轮到对方队长操作。|It is the opposing captain’s turn.
你尚未在该房间落座。|You have not taken a seat in this room.
只有主办方可以执行此操作。|Only the organizer can do this.
选择和禁用的项目不能相同。|You cannot pick and ban the same game.
只有已落座的选手可以提交比赛问题。|Only seated players can report match issues.
该项目暂不可用，请联系主办方。|This game is unavailable. Contact the organizer.
你的本场项目已完成。|You have already finished this game.
项目池中找不到所选项目。|The selected game is not in the pool.
项目池中已没有足够的可选项目。|Not enough available games remain in the pool.
所选项目当前不可用，请重新选择。|This game is unavailable. Choose another.
请等待双方出战选手和队长全部准备。|Wait for both players and captains to ready up.
当前不能修改准备状态。|Readiness cannot be changed now.
只有主办方或裁判可以执行此操作。|Only an organizer or referee can do this.
本队已确认该场结果。|Your team has already confirmed this result.
请等待双方队长确认恢复比赛。|Wait for both captains to confirm resuming play.
抽签开始后不能关闭房间。|Rooms cannot be closed after the draw starts.
暂时无法分配房间码，请稍后重试。|Unable to allocate a room code. Try again later.
该房间码已被使用。|This room code is already in use.
找不到该比赛房间，请检查房间码。|Room not found. Check the room code.
你已被移出此房间，请联系房主或赛事管理员。|You were removed from this room. Contact its owner or an event administrator.
仅赛事管理员或该房间的房主可以管理人员。|Only event administrators or this room’s owner can manage participants.
不能移出自己。|You cannot remove yourself.
参赛人员已被移出，需房主或赛事管理员处理后继续。|A player was removed. The owner or an event administrator must resolve this before play continues.
仅当对方本局已完赛且未认输时，才可认输。|You can concede only after your opponent has finished without conceding.
本局休整尚未结束，请等待 30 秒展示完毕。|Intermission is still active. Wait for the 30-second results display to finish.
该席位已有人落座。|This seat is occupied.
该房间已停止选座。|Seating is closed in this room.
六个选手席位尚未全部坐满。|Not all six seats are occupied.
队长准备后，席位已锁定。|Seats are locked after a captain readies up.
比赛阶段已变化，请刷新房间后重试。|The phase has changed. Refresh the room and retry.
比赛结果已变化，请重新确认。|The result has changed. Confirm it again.
对局状态过大，无法同步。|The game state is too large to sync.
本队包干时间已用尽，正在结算比赛。|Your team’s time bank is exhausted. Settling the match.
找不到该试玩项目。|Practice game not found.
找不到该用户或用户已停用。|User not found or deactivated.
没有执行此操作的权限。|You do not have permission to do this.
找不到对应内容，请检查后重试。|Content not found. Check and retry.
房间状态已变化，请刷新后重试。|The room state has changed. Refresh and retry.
提交的内容无效，请检查后重试。|Invalid submission. Check and retry.
服务暂时不可用，请稍后重试。|Service temporarily unavailable. Try again later.
网络连接失败，请检查网络后重试。|Connection failed. Check your network and retry.
操作失败，请稍后重试。|Action failed. Try again later.
`;
export const errorTranslations=Object.fromEntries(entries.trim().split('\n').map(line=>{const i=line.indexOf('|');return[line.slice(0,i),line.slice(i+1)];}));
