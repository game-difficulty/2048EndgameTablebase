# 主站账号设置同步独立发布（2026-09-25）

目标：仅在 `https://2048tables.online/` 发布账号设置同步，不替换 Play 或直播入口。

发布包以线上主站源码为基底，只叠加以下改动：

- `backend/auth/db.py`：初始化 `user_preferences` 表。
- `backend/profile/routes.py`、`backend/profile/preferences.py`：账号偏好读取与保存接口。
- 主站 `App.vue`、`useAppSettings.js`、设置页和设置会话、`accountPreferences.js`：登录后读取、修改后保存、跨页面应用账号偏好，以及失败重试提示。

构建出的主站 `index.html`、内容寻址的 `assets/` 和兼容 CSS 已上传；原有资源文件保留。`live/index.html`、`human/index.html`、Play 独立目录和 Nginx 配置未替换。线上回滚副本位于 `/opt/2048tables/backups/main-settings-20260925/`。回滚时恢复该目录下的 `auth-db.py`、`profile-routes.py`、`index.html`、`index.html.gz`，移除新加的 `backend/profile/preferences.py`，重启 `2048tables-cloud.service`；新静态资源可留存，因为旧入口不会引用。

验证：前端独立构建通过；账号偏好前端 2 项测试、后端 4 项测试通过；线上主站和 Play 的 `/api/profile/preferences` 对未登录请求均返回 401；主站、Play、直播页均返回 200；主站入口引用新资源，静态资源 CDN 返回 HIT；主站与 Play 服务均处于 active。线上账号未用于修改偏好的端到端测试。
