const entries=`
首个特殊块还剩|Until first cargo
你自己|You
暂无已落座选手|No seated players yet
输入用户 ID|Enter user ID
已移出人员|Removed participants
导入方式|Import mode
当前用户名|Current username
用户 ID|User ID
队名（可空）,外援0或1,队长0或1,队内序号（团队对战填1/2/3）。|team name (optional), guest 0/1, captain 0/1, team position (1/2/3 for team matches).
每行：当前用户名或用户ID（按导入方式选择）,队名（未分组留空）,外援0或1,队长0或1,队内序号（团队对战填1/2/3）。仅填用户名或 ID 也可导入。用户名按主站规则匹配当前有效账号，不匹配历史用户名；任一行匹配失败则整批不导入。保存将整体替换当前名单，并取消旧邀请；锁定参赛人员后只能调整同一批人员的分组。自由组队的分组须各指定一名队长，导入后仍需队长提交。|Each row: current username or user ID (according to import mode), team name (blank if unassigned), guest 0/1, captain 0/1, team position (1/2/3 for team matches). Usernames or IDs alone are accepted. Usernames match current active accounts using the main site's rules, not historical names. Any unmatched row blocks the entire import. Saving replaces the entire roster and cancels old invitations. Once participants are locked, only team assignments may change. Self-organized teams need one captain each and still require captain submission after import.
跟随主站语言，仅同步到本站|Follow the main-site language (one-way sync)
同步中…|Syncing…
同步登录|Sync sign-in
按得分选最佳五局，不足五局以零补齐，再以五局平均盘面和计算个人 rating；零有效局时 rating 为零。团队盘面和与 rating 分别为队员值之和。|The five highest-scoring games are selected; missing games count as zero. Individual rating is calculated from their average tile sum, or zero if there are no eligible games. Team tile sum and rating are the sums of the individual values.
三人团队选 Ban|Three-player team pick/ban
三人团队 · 选 Ban 对战|Three-player teams · Pick/ban matches
3×3 · 四队积分统计|3×3 · Four-team statistics
统计赛事|Statistics event
选择赛事，查看规则与比赛安排。|Choose an event to view its rules and schedule.
项目练习 →|Practice games →
赛事状态|Event status
查看赛事 →|View event →
暂无此状态的赛事。|No events with this status.
创建赛事|Create event
赛事名称|Event name
赛事地址标识|Event URL slug
例如 summer-cup-1|e.g. summer-cup-1
简介|Description
规则与公告|Rules and announcements
当前创建三人团队选 Ban 赛事。默认关闭报名，可在赛事管理中设置报名方式并开放。|Creates a three-player team pick/ban event. Registration is closed by default; configure and open it in event management.
← 全部赛事|← All events
正在加载赛事…|Loading event…
参赛信息|Participation
报名、邀请与队伍提交请在下方操作，以举办方公布的报名状态为准。|Register, manage invitations, and submit your team below. Availability follows the organizer’s registration settings.
请按报名表队内序号落座。开战后超过 15 分钟，仅一方就位则该方 3:0 获胜；双方均未就位则 0:0。|Use your registered team position. After the 15-minute grace period, a ready team wins 3:0 against an unready team; if neither is ready, the result is 0:0.
前往直播大厅 →|Live lobby →
参赛方式|How to participate
使用已登记的 Table 账号，在对局站进行 3×3 对局。无需进入比赛房间。|Play 3×3 games on the Play site using your registered Table account. No match room is needed.
仅统计比赛期间开始并完成的有效对局，外站导入局不计入。|Only eligible games started and finished during the event count. Games imported from other sites are excluded.
前往对局站 →|Play site →
指定赛事举办方|Assign an organizer
当前举办方：|Current organizer:
举办方 Table 用户 ID|Organizer’s Table user ID
核对用户|Look up user
该用户将能够编辑本赛事信息、导入分组名单及管理报名。|This user will be able to edit the event, import rosters, and manage registration.
确认指定为举办方|Confirm organizer
赛事纪录候选成绩|Record-eligible results
仅列入本轮获胜队伍的有效完赛成绩，按项目及规则版本分别评选。|Only valid completed results from the winning team qualify, grouped by game and rules version.
· 选手 ID|· Player ID
赛程与结果|Schedule and results
场 · 北京时间|matches · China Standard Time
进入房间 →|Enter room →
举办方尚未关联比赛房间。|The organizer has not linked any match rooms.
赛事管理 · 房间关联|Event management · Linked rooms
已有房间不会自动归入赛事。关联不会改变原房间地址、项目规则或比赛进度。|Existing rooms are not linked automatically. Linking does not change a room’s URL, rules, or progress.
已有房间码|Existing room code
关联房间|Link room
为本赛事创建房间|Create a room for this event
编辑赛事信息|Edit event
名称|Name
保存赛事信息|Save event
请输入有效的 Table 用户 ID。|Enter a valid Table user ID.
举办方已更新。|Organizer updated.
筹备中|Upcoming
进行中|In progress
已结束|Finished
赛事信息已保存。|Event saved.
全部赛事|All events
进入赛事查看详情。|Open the event for details.
名单导入 · 成绩统计|Roster import · Statistics
举办方尚未发布规则。|The organizer has not published the rules yet.
尚未指定|Not assigned
时间待定|Time to be confirmed
统计型赛事状态按时间窗自动切换；编辑公告不会改变统计时间或成绩算法。|Statistics events change status automatically according to their time window. Editing announcements does not change the window or scoring algorithm.
赛事状态用于目录展示，不会启动、停止或修改已有对局。修改公告不会改变房间内已固定的项目规则。|Event status affects the directory only; it does not start, stop, or modify matches. Announcement edits do not change locked room rules.
报名与参赛名单|Registration and roster
刷新|Refresh
人 ·|players ·
登录 Table 账号后报名或处理邀请 →|Sign in to Table to register or manage invitations →
你的 Table 用户 ID：|Your Table user ID:
可将此 ID 提供给队长用于邀请。|Share this ID with your captain to receive an invitation.
你已|You are
报名参赛|Register
退出报名|Withdraw registration
队伍名称|Team name
创建队伍并报名|Create team and register
」邀请你加入|” invited you to join
接受|Accept
拒绝|Decline
我的队伍 ·|My team ·
队员 Table 用户 ID|Player’s Table user ID
发出邀请|Send invitation
取消邀请|Cancel invitation
解散队伍|Dissolve team
离开队伍（保留个人报名）|Leave team (keep individual registration)
暂无报名或导入名单。|No registrations or imported roster yet.
举办方 · 报名设置、导入与锁定|Organizer · Registration, import and locking
报名方式|Registration mode
人数上限（0 不限）|Player limit (0 = unlimited)
开放报名|Open registration
保存设置|Save settings
锁定参赛人员|Lock participants
锁定最终名单|Lock final roster
解锁原因|Reason for unlocking
解锁（不会自动开放报名）|Unlock (does not reopen registration)
导入 / 调整名单|Import / edit roster
每行：用户ID,队名（未分组留空）,外援0或1,队长0或1,队内序号（团队对战填1/2/3）。仅有用户 ID 也可导入。保存将整体替换当前名单，并取消旧邀请；锁定参赛人员后只能调整同一批人员的分组。自由组队的分组须各指定一名队长，导入后仍需队长提交。|Each row: user ID, team name (blank if unassigned), guest 0/1, captain 0/1, team position (1/2/3 for team matches). User IDs alone are accepted. Saving replaces the entire roster and cancels previous invitations. Once participants are locked, only their team assignments may change. Self-organized teams need one captain each and still require captain submission after import.
载入当前名单编辑|Load current roster for editing
名单|Roster
校验并预览|Validate and preview
预览 · 尚未保存|Preview · Not saved
确认替换为这|Replace roster with these
位选手|players
最近操作记录|Recent activity
· 管理/操作账号|· Acting account
单人报名|Individual registration
自由组队报名|Self-organized teams
个人报名 · 举办方分队|Individual entry · Organizer assigns teams
确定退出报名？|Withdraw your registration?
解散后所有队员仍保留报名，但会变为未组队。确定继续？|All players will remain registered but become unassigned. Dissolve this team?
锁定参赛人员后不能自主报名或退出；举办方仍可调整分组。确定？|Lock participants? Players cannot join or withdraw afterwards, but the organizer can still change team assignments.
锁定最终名单后，报名、组队和导入均停止。确定名单已经核对完毕？|Lock the final roster? Registration, team changes, and imports will stop. Please confirm the roster is correct.
操作已保存。|Changes saved.
请提供 1–500 位选手。|Provide 1–500 players.
名单已保存。未分组的选手仍为待分组状态。|Roster saved. Unassigned players remain pending team assignment.
最终名单已锁定|Final roster locked
参赛人员已锁定，分组可调整|Participants locked; team assignments editable
报名开放|Registration open
报名未开放|Registration closed
由举办方登记|Registered by organizer
报名|registered
单人参赛|Individual entry
待分组 / 待组队|Awaiting team assignment
已提交报名|Registration submitted
待队长提交|Awaiting captain submission
撤回队伍报名|Withdraw team submission
全员确认，提交队伍报名|All confirmed · Submit team
已锁定|Locked
分组草案|Draft teams
· 队长|· Captain
· 外援|· Guest player
参赛选手|Players
待分组选手|Unassigned players
（外援）|(Guest player)
未分组|Unassigned
赛事进程与成绩|Progress and results
以下为上次成功读取的数据。|Showing the last successfully loaded data.
· 更新于|· Updated
目前尚未分组。|Teams have not been assigned yet.
位选手待分组，个人成绩照常统计；团队成绩在分组后汇总。|players await team assignment. Individual results are tracked now; team totals will appear after assignment.
盘面和合计|Total tile sum
Rating 合计|Total rating
已入选|Selected
/25 局 · 有效完赛|/25 games · Eligible completed games
局|games
/5 局|/5 games
个人成绩与最佳五局|Individual results and top five games
按个人 rating 排列。分数决定入选局；不足五局按零补足后计算。* 表示尚未完成五局。|Ranked by individual rating. Games are selected by score; missing games count as zero. * indicates fewer than five completed games.
盘面和|Tile sum
得分|Score
开始时间（北京时间）|Started (China Standard Time)
完成时间|Finished
暂无符合时间和有效性要求的已完成局。|No completed games meet the time and eligibility requirements yet.
更新中…|Updating…
刷新成绩|Refresh results
尚未开赛|Not started
比赛进行中|Event in progress
统计时间窗已结束|Statistics window closed
全员满五局|All players have five games
成绩未满|Incomplete results
待分组|Unassigned
赛程与锁定名单|Schedule and locked roster
自由房间：任意已登录选手可落座。正式赛程请先锁定报名名单及队内序号。|Open rooms allow any signed-in player to sit. For an official fixture, lock the roster and team positions first.
当前对战房间需要三人团队名单；单人赛和统计赛不使用此对战流程。|These rooms require three-player teams. Individual and statistics events do not use this match flow.
使用锁定队伍创建赛程房间|Create a scheduled room using locked teams
黄方队伍|Yellow team
请选择|Select
白方队伍|White team
预定开战时间（本设备时区）|Scheduled start (device time zone)
未绑定队伍时为自由房间，不计入正式赛事纪录。设置开战时间后，到点方可开始抽签；超过15分钟，已全员落座且队长准备的一方3:0获胜，双方均未就位则0:0。|Unassigned rooms are open rooms and do not count toward official records. The draw cannot start before the scheduled time. After 15 minutes, a fully seated team with its captain ready wins 3:0 against an unready team; if neither is ready, the result is 0:0.
无法读取锁定名单，请重新选择赛事。|Could not load the locked roster. Select the event again.
2048 赛事项目试玩|2048 Tournament Practice
比赛项目试玩|Tournament Practice
以下页面用于举办方验收规则、选手熟悉操作。试玩成绩不会进入正式比赛。|Try the tournament games and familiarize yourself with the controls. Practice results do not count toward official matches.
开始试玩 →|Play →
← 全部项目|← All games
单人试玩|Solo practice
步数|Moves
方块数量|Tile count
距下次轮换|Until next rotation
数量|Count
掷出|Rolled
点|on the die
本次试玩结果|Practice result
关闭结果浮窗|Close results
再试一次|Try again
撤销一步|Undo
方向键 / WASD / 滑动操作|Arrow keys / WASD / Swipe
试玩排行榜|Practice leaderboard
试玩榜|Practice leaderboard
仅供试玩|Practice only
正在加载…|Loading…
游客可查看；|Guests can view results;
后记录个人最佳。|to save your personal best.
我的最佳：|Personal best:
暂无记录。|No records yet.
玩法说明|How to play
棋盘|Board
结算|Scoring
镜面棋盘怎么走？|How does the mirror board work?
把中央十字想成真正的墙。向左滑出最左边的砖会从最右边回来；上下同理。砖最终都停在中央墙的两侧。|The central cross is a wall. Tiles leaving the left edge re-enter on the right; top and bottom connect in the same way. Tiles stop on either side of the central walls.
WASM 暂不可用，本次已用确定性随机出数代替：|WASM is unavailable. This run uses deterministic random spawning instead:
切换为浅色模式|Switch to light mode
切换为深色模式|Switch to dark mode
浅色|Light
深色|Dark
正在同步登录状态|Checking sign-in
已登录|Signed in
游客|Guest
已送出|Delivered
方块超限|Tile limit exceeded
完成目标|Target reached
超过12块，本次结束|More than 12 tiles · Game over
本次试玩结束|Practice finished
运输结束|Transport finished
角位放置墙|Wall placed in a corner
边位放置墙|Wall placed on an edge
中心位放置墙|Wall placed in the center
无路可走时结束，按送出数量比较|Ends with no legal moves; compare deliveries
先达到目标者获胜|First to reach the target wins
双方死亡后比较盘面和|Compare tile sums after both boards have no legal moves
双方死亡后比较得分|Compare scores after both boards have no legal moves
榜单暂不可用，不影响试玩。|Leaderboard unavailable. You can still practice.
正在记录成绩…|Saving result…
个人最佳已更新。|Personal best updated.
本次未超过个人最佳。|Your personal best is unchanged.
登录已过期，本次未记录。|Sign-in expired. This result was not saved.
成绩记录失败，不影响试玩。|Could not save the result. You can still practice.
入口|Entrance
出口|Exit
出口 ↓|Exit ↓
待测|TBD
随机 · 12格|Random · 12 cells
赛事状态按统计时间窗自动切换；只统计期间开始并完成、符合对局站有效性要求的原生局，外站导入局不计入。|Event status follows the scoring window automatically. Only eligible native games started and completed within that window count; imported games are excluded.
等待报名或举办方导入名单。目前尚未分组，不展示虚构队伍或成绩。|Waiting for registration or an organizer-imported roster. No teams have been assigned yet; no placeholder teams or results are shown.
（北京时间） · 页面可见时每 30 秒刷新|(Beijing time) · Refreshes every 30 seconds while this page is visible
`;
export const extraMessages=Object.fromEntries(entries.trim().split('\n').map(line=>{const i=line.indexOf('|');return[line.slice(0,i),line.slice(i+1)];}));
