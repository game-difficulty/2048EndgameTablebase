from __future__ import annotations

import html
import json


def build_startup_error_page_html(
    app_title: str,
    language: str,
    title: str,
    message: str,
) -> str:
    escaped_title = html.escape(title)
    escaped_message = html.escape(message)
    escaped_language = html.escape(language, quote=True)
    escaped_app_title = html.escape(app_title)
    return f"""<!DOCTYPE html>
<html lang="{escaped_language}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_app_title} - {escaped_title}</title>
  <style>
    :root {{
      color-scheme: light dark;
      font-family: "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei UI", "Microsoft YaHei", "WenQuanYi Micro Hei", "Segoe UI Symbol", "Noto Color Emoji", "Segoe UI", sans-serif;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #f5f7fb 0%, #e9edf5 100%);
      color: #111827;
      display: grid;
      place-items: center;
      padding: 24px;
    }}
    main {{
      width: min(820px, 100%);
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 20px;
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.14);
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 30px;
    }}
    h2 {{
      margin: 0 0 18px;
      font-size: 18px;
      color: #b91c1c;
    }}
    pre {{
      margin: 0;
      padding: 18px;
      border-radius: 14px;
      background: #0f172a;
      color: #e5e7eb;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font: 14px/1.65 "Cascadia Mono", "Consolas", monospace;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escaped_app_title}</h1>
    <h2>{escaped_title}</h2>
    <pre>{escaped_message}</pre>
  </main>
</body>
</html>
"""


def build_startup_page_html(
    app_title: str,
    language: str,
    is_dark_mode: bool,
    message: str,
    is_error: bool = False,
    *,
    title: str | None = None,
    badge: str | None = None,
    show_spinner: bool | None = None,
    action_label: str | None = None,
    action_url: str | None = None,
    manual_link_hint: str | None = None,
) -> str:
    escaped_message = html.escape(message).replace("\n", "<br>")
    accent = (
        "#ff8e72"
        if is_error and is_dark_mode
        else "#b93818"
        if is_error
        else "#6fe0c2"
        if is_dark_mode
        else "#1f6f5f"
    )
    color_scheme = "dark" if is_dark_mode else "light"
    show_spinner = (not is_error) if show_spinner is None else show_spinner
    if is_dark_mode:
        background = (
            "radial-gradient(circle at top, #402018 0%, #251918 44%, #111315 100%)"
            if is_error
            else "radial-gradient(circle at top, #14332f 0%, #1b2025 42%, #101214 100%)"
        )
        surface = "rgba(20, 23, 27, 0.84)"
        text = "#f3f4f6"
        muted = "#aab3bd"
        border = "rgba(255, 255, 255, 0.10)"
        panel_shadow = "0 28px 80px rgba(0, 0, 0, 0.40)"
        badge_background = "rgba(255, 255, 255, 0.08)"
        badge_border = "rgba(255, 255, 255, 0.10)"
        spinner_border = "rgba(255, 255, 255, 0.14)"
        action_background = "rgba(111, 224, 194, 0.16)"
        action_text = "#d9fff6"
        action_border = "rgba(111, 224, 194, 0.28)"
        link_color = "#8ff6d8"
        link_note_color = "#cfd6df"
    else:
        background = (
            "radial-gradient(circle at top, #fff4ec 0%, #f4eee5 42%, #e9e2d8 100%)"
            if is_error
            else "radial-gradient(circle at top, #f5fbf7 0%, #ece9dd 42%, #e0d8cb 100%)"
        )
        surface = "rgba(255, 255, 255, 0.84)"
        text = "#1d1d1f"
        muted = "#5f6368"
        border = "rgba(0, 0, 0, 0.08)"
        panel_shadow = "0 28px 80px rgba(45, 35, 24, 0.14)"
        badge_background = "rgba(255, 255, 255, 0.72)"
        badge_border = "rgba(0, 0, 0, 0.06)"
        spinner_border = "rgba(0, 0, 0, 0.08)"
        action_background = "rgba(31, 111, 95, 0.12)"
        action_text = "#12453b"
        action_border = "rgba(31, 111, 95, 0.16)"
        link_color = "#0f766e"
        link_note_color = "#4b5563"

    link_markup = ""
    if action_label and action_url:
        escaped_url = html.escape(action_url, quote=True)
        escaped_label = html.escape(action_label)
        hint_markup = ""
        if manual_link_hint:
            escaped_hint = html.escape(manual_link_hint).replace("\n", "<br>")
            hint_markup = (
                f"<p class=\"link-note\">{escaped_hint}<br>"
                f"<a class=\"inline-link\" href=\"{escaped_url}\" target=\"_blank\" "
                f"rel=\"noreferrer noopener\" onclick='return openExternalLink({json.dumps(action_url)})'>"
                f"{escaped_url}</a></p>"
            )
        link_markup = (
            "<div class=\"actions\">"
            f"<a class=\"action-link\" href=\"{escaped_url}\" target=\"_blank\" "
            f"rel=\"noreferrer noopener\" onclick='return openExternalLink({json.dumps(action_url)})'>"
            f"{escaped_label}</a>"
            f"{hint_markup}"
            "</div>"
        )

    return f"""<!DOCTYPE html>
<html lang=\"{html.escape(language, quote=True)}\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(app_title)}</title>
  <script>
    function openExternalLink(url) {{
      try {{
        if (
          window.pywebview &&
          window.pywebview.api &&
          typeof window.pywebview.api.open_external_url === 'function'
        ) {{
          window.pywebview.api.open_external_url(url);
          return false;
        }}
      }} catch (error) {{
      }}
      return true;
    }}
  </script>
  <style>
    :root {{
      color-scheme: {color_scheme};
      font-family: \"Noto Sans CJK SC\", \"Noto Sans SC\", \"Source Han Sans SC\", \"PingFang SC\", \"Hiragino Sans GB\", \"Microsoft YaHei UI\", \"Microsoft YaHei\", \"WenQuanYi Micro Hei\", \"Segoe UI Symbol\", \"Noto Color Emoji\", \"Segoe UI\", sans-serif;
      --accent: {accent};
      --surface: {surface};
      --text: {text};
      --muted: {muted};
      --border: {border};
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      background: {background};
      color: var(--text);
    }}
    .panel {{
      width: min(560px, calc(100vw - 48px));
      padding: 32px 30px;
      border-radius: 24px;
      background: var(--surface);
      border: 1px solid var(--border);
      box-shadow: {panel_shadow};
      backdrop-filter: blur(18px);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      background: {badge_background};
      border: 1px solid {badge_border};
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 18px 0 8px;
      font-size: 34px;
      line-height: 1.1;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 19px;
      line-height: 1.35;
      font-weight: 700;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.7;
      word-break: break-word;
    }}
    .actions {{
      display: grid;
      gap: 14px;
      margin-top: 24px;
    }}
    .action-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 0 18px;
      border-radius: 14px;
      background: {action_background};
      border: 1px solid {action_border};
      color: {action_text};
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
    }}
    .inline-link {{
      color: {link_color};
      text-decoration: none;
      word-break: break-all;
      font-weight: 600;
    }}
    .link-note {{
      color: {link_note_color};
      font-size: 13px;
      line-height: 1.6;
    }}
    .spinner {{
      width: 18px;
      height: 18px;
      margin-top: 22px;
      border-radius: 50%;
      border: 2px solid {spinner_border};
      border-top-color: var(--accent);
      animation: spin 0.9s linear infinite;
      display: {("block" if show_spinner else "none")};
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
  </style>
</head>
<body>
  <main class=\"panel\">
    <div class=\"badge\">{badge}</div>
    <h1>{html.escape(app_title)}</h1>
    <h2>{html.escape(title)}</h2>
    <p>{escaped_message}</p>
    {link_markup}
    <div class=\"spinner\" aria-hidden=\"true\"></div>
  </main>
</body>
</html>
"""

