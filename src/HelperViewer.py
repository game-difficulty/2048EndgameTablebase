import html
import os
import re
import sys
import threading
import traceback
import webbrowser
from http.server import SimpleHTTPRequestHandler, HTTPServer

import markdown
import markdown.extensions.extra
import markdown.extensions.sane_lists
import markdown.extensions.nl2br
import markdown.extensions.toc
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QApplication, QListWidget, QVBoxLayout, QWidget, QSplitter, QTextBrowser

from Config import SingletonConfig, logger


class WebColorManager:
    def get_css_color(self, index):
        theme = SingletonConfig().config.get('theme', 'Default')

        if theme == 'Dark':
            colors = {
                1: '#1e1e1e',  # 主背景色
                3: '#252526',  # 侧边栏/代码块背景
                6: '#333333',  # 输入框背景
                10: '#d4d4d4', # 主文本颜色
                11: '#58a6ff', # 目录链接颜色 (适合暗色的浅蓝色)
                12: '#37373d', # 目录 hover 背景色 (深灰色)
                13: '#8b949e'  # 次级文本颜色 (如4级标题的柔和灰色)
            }
            fallback_color = '#ffffff'

        else:
            # === 亮色配色方案 (保留你原有的设定) ===
            colors = {
                1: '#ffffff',  # 主背景色 (纯白)
                3: '#f6f8fa',  # 侧边栏/代码块背景 (GitHub 风格的极浅灰蓝)
                6: '#ffffff',  # 输入框背景
                10: '#24292e',  # 主文本颜色 (接近纯黑的深灰)
                11: '#0366d6',  # 目录链接颜色 (经典深蓝色)
                12: '#eaecef',  # 目录 hover 背景色 (浅灰色)
                13: '#586069'  # 次级文本颜色
            }
            fallback_color = '#000000'  # 亮色模式下回退为黑色

        return colors.get(index, fallback_color)


def get_resource_path(relative_path):
    """ 获取资源绝对路径，多重后备机制，解决打包后的路径玄学问题 """
    base_paths = []

    if getattr(sys, 'frozen', False):
        # 如果是 PyInstaller 打包环境
        base_paths.append(sys._MEIPASS)  # PyInstaller 的临时/依赖目录 (如 _internal)
        base_paths.append(os.path.dirname(sys.executable))  # exe 所在的同级目录
    else:
        # 源代码运行环境
        base_paths.append(os.path.abspath("."))

    # 依次遍历可能的目录，哪个存在就返回哪个
    for base_path in base_paths:
        full_path = os.path.join(base_path, relative_path)
        if os.path.exists(full_path):
            return full_path

    # 如果全都没找到，返回默认期望路径，让后续代码抛出 FileNotFoundError
    return os.path.join(getattr(sys, '_MEIPASS', os.path.abspath(".")), relative_path)


class HelpServer(HTTPServer):
    """自定义 HTTP Server，用于在特定路由返回动态生成的 HTML"""

    def __init__(self, server_address, RequestHandlerClass, md_file):
        super().__init__(server_address, RequestHandlerClass)
        self.md_file = md_file


class HelpRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # 强制指定静态资源目录，解决打包后找不到本地依赖的问题
        super().__init__(*args, directory=get_resource_path("."), **kwargs)

    def log_message(self, format, *args):
        """
        覆盖父类的日志方法。
        阻止底层偷偷向 sys.stderr 写数据，彻底解决 ERR_EMPTY_RESPONSE 问题。
        """
        pass

    def do_GET(self):
        if self.path == '/' or self.path == '/help':
            try:
                # 只有成功生成了 HTML，才发送 200 OK 状态码
                html_content = self.generate_html()
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_content.encode('utf-8'))
            except Exception as e:
                # 如果崩溃了，把崩溃日志发送给浏览器
                error_trace = traceback.format_exc()
                logger.error(f"Render HTML failed:\n{error_trace}")

                error_html = f"<h1>Internal Server Error</h1><pre>{error_trace}</pre>"
                self.send_response(500)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(error_html.encode('utf-8'))
        else:
            super().do_GET()

    def generate_html(self):
        # 动态读取语言配置
        lang = SingletonConfig().config.get('language', 'en')
        md_file = 'helpZH.md' if lang == 'zh' else 'help.md'

        md_file_path = get_resource_path(md_file)

        if not os.path.exists(md_file_path):
            return f"<h1>Error: File not found</h1><p>Looking for: {md_file_path}</p>"

        with open(md_file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()

        # 如果前面的 import 没起作用，这里依然会用显式声明的方式加载
        extensions = ['extra', 'sane_lists', 'nl2br', 'toc']
        html_body = markdown.markdown(md_content, extensions=extensions)

        toc_html = self.extract_headers_to_toc(html_body)

        color_mgr = WebColorManager()
        bg_color = color_mgr.get_css_color(1)
        text_color = color_mgr.get_css_color(10)
        sidebar_bg = color_mgr.get_css_color(3)

        toc_link_color = color_mgr.get_css_color(11)
        toc_hover_bg = color_mgr.get_css_color(12)
        sub_text_color = color_mgr.get_css_color(13)

        # 构建完整的 HTML，包含左右分栏布局
        full_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Help Document</title>

            <script>
            window.MathJax = {{
                tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true, processEnvironments: true }},
                svg: {{ fontCache: 'global' }},
                options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
            }};
            </script>
            <script id="MathJax-script" async src="/mathjax/tex-svg.js"></script>

            <style>
                /* 全局重置 */
                body, html {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}

                /* 核心布局：使用 Flexbox 实现侧边栏和主内容的拆分 */
                .container {{ display: flex; height: 100vh; }}

                /* 左侧目录栏样式 */
                .sidebar {{
                    width: 300px;
                    min-width: 200px;
                    background-color: {sidebar_bg};
                    border-right: 1px solid #e1e4e8;
                    overflow-y: auto;
                    padding: 20px 0;
                    box-sizing: border-box;
                }}

                .sidebar ul {{ list-style: none; padding-left: 0; margin: 0; }}
                .sidebar li {{ margin: 0; }}
                .sidebar a {{
                    display: block;
                    padding: 6px 20px;
                    color: {toc_link_color};
                    text-decoration: none;
                    font-size: 14px;
                    line-height: 1.5;
                }}
                .sidebar a:hover {{ background-color: {toc_hover_bg}; text-decoration: underline; }}

                /* 针对不同级别的标题添加缩进 */
                .sidebar .toc-level-1 {{ padding-left: 20px; font-weight: bold; }}
                .sidebar .toc-level-2 {{ padding-left: 35px; }}
                .sidebar .toc-level-3 {{ padding-left: 50px; font-size: 13px; }}
                .sidebar .toc-level-4 {{ padding-left: 65px; font-size: 13px; color: {sub_text_color}; }}

                /* 右侧主体内容样式 */
                .content-wrapper {{
                    flex-grow: 1;
                    overflow-y: auto;
                    padding: 40px;
                    background-color: {bg_color};
                    scroll-behavior: smooth;
                }}

                .markdown-body {{
                    max-width: 900px;
                    margin: 0 auto;
                    color: {text_color};
                    line-height: 1.6;
                }}

                /* Markdown 元素的基础样式 (保留你之前的设定) */
                .markdown-body h1, .markdown-body h2 {{ border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 24px; margin-bottom: 16px; font-weight: 600; }}
                .markdown-body ul {{ list-style-type: disc; padding-left: 2em; }}
                .markdown-body ol {{ list-style-type: decimal; padding-left: 2em; }}
                .markdown-body pre {{ background-color: rgba(0,0,0,0.05); border-radius: 6px; padding: 16px; overflow: auto; font-size: 85%; }}
                .markdown-body code {{ font-family: monospace; padding: 0.2em 0.4em; background-color: rgba(27,31,35,0.05); border-radius: 3px; font-size: 85%; }}
                .markdown-body pre code {{ padding: 0; background-color: transparent; }}
                .markdown-body table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
                .markdown-body th, .markdown-body td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
                .markdown-body tr:nth-child(even) {{ background-color: rgba(0,0,0,0.03); }}

                /* 滚动条美化 */
                ::-webkit-scrollbar {{ width: 8px; }}
                ::-webkit-scrollbar-track {{ background: transparent; }}
                ::-webkit-scrollbar-thumb {{ background: #c1c1c1; border-radius: 4px; }}
                ::-webkit-scrollbar-thumb:hover {{ background: #a8a8a8; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="sidebar">
                    {toc_html}
                </div>

                <div class="content-wrapper" id="content-scroll">
                    <div class="markdown-body">
                        {html_body}
                    </div>
                </div>
            </div>

            <script>
                // 优化：点击目录栏后，平滑滚动到对应的锚点，并考虑顶部 padding
                document.querySelectorAll('.sidebar a').forEach(anchor => {{
                    anchor.addEventListener('click', function (e) {{
                        e.preventDefault();
                        const targetId = this.getAttribute('href').substring(1);
                        const targetElement = document.getElementById(targetId);
                        const scrollContainer = document.getElementById('content-scroll');

                        if (targetElement && scrollContainer) {{
                            // 计算元素相对于滚动容器的位置
                            const topPos = targetElement.offsetTop - scrollContainer.offsetTop - 20;
                            scrollContainer.scrollTo({{ top: topPos, behavior: 'smooth' }});
                        }}
                    }});
                }});
            </script>
        </body>
        </html>
        """
        return full_html

    def extract_headers_to_toc(self, html_body):
        """将 HTML 中的标题提取出来，组装成侧边栏的 <ul> 列表"""
        pattern = re.compile(r'<h([1-6]) id="(.*?)">(.*?)</h\1>')
        headers = pattern.findall(html_body)

        toc_lines = ['<ul>']
        for level, anchor_id, text in headers:
            clean_text = re.sub(r'<.*?>', '', text).strip()
            final_text = html.unescape(clean_text)
            # 根据 h1, h2, h3 赋予不同的 class 以控制 CSS 缩进
            toc_lines.append(f'<li><a href="#{anchor_id}" class="toc-level-{level}">{final_text}</a></li>')
        toc_lines.append('</ul>')

        return '\n'.join(toc_lines)


class FallbackMDViewer(QMainWindow):
    def __init__(self, md_file):
        super().__init__()
        self.md_file = md_file
        self.toc_anchors = []
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle(self.tr("Help (Basic Mode)"))
        self.setWindowIcon(QIcon(r"pic\2048_2.ico"))
        self.resize(1000, 700)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧目录
        self.toc_list = QListWidget()
        self.toc_list.setMaximumWidth(300)
        self.toc_list.currentRowChanged.connect(self.onTOCClicked)  # type: ignore

        # 右侧基础富文本浏览器
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)  # 允许点击外部链接打开系统浏览器

        splitter.addWidget(self.toc_list)
        splitter.addWidget(self.browser)
        splitter.setSizes([250, 750])

        layout.addWidget(splitter)

        self.loadMarkdown()

    def loadMarkdown(self):
        try:
            with open(self.md_file, 'r', encoding='utf-8') as file:
                md_content = file.read()
        except FileNotFoundError:
            self.browser.setHtml(f"<h1>Error: {self.md_file} not found.</h1>")
            return

        # Markdown 转换为基础 HTML
        extensions = ['extra', 'sane_lists', 'nl2br', 'toc']
        html_body = markdown.markdown(md_content, extensions=extensions)

        color_mgr = WebColorManager()
        bg_color = color_mgr.get_css_color(1)
        text_color = color_mgr.get_css_color(10)

        # 针对 QTextBrowser 的极简 CSS (不支持 flex, 只能用基础样式)
        simple_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: "Segoe UI", Arial, sans-serif; color: {text_color}; background-color: {bg_color}; padding: 10px; line-height: 1.5; }}
                h1, h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #aaa; padding: 5px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; font-family: monospace; }}
                code {{ background-color: #f4f4f4; font-family: monospace; }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """
        self.browser.setHtml(simple_html)
        self.extractHeaders(html_body)

    def extractHeaders(self, html_body):
        pattern = re.compile(r'<h([1-6]) id="(.*?)">(.*?)</h\1>')
        headers = pattern.findall(html_body)

        self.toc_anchors.clear()
        self.toc_list.clear()

        for level, anchor_id, text in headers:
            clean_text = re.sub(r'<.*?>', '', text).strip()
            final_text = html.unescape(clean_text)
            self.toc_anchors.append(anchor_id)
            self.toc_list.addItem('    ' * (int(level) - 1) + final_text)

    def onTOCClicked(self, index):
        if 0 <= index < len(self.toc_anchors):
            anchor_id = self.toc_anchors[index]
            # 核心跳转逻辑：QTextBrowser 内置的支持
            self.browser.scrollToAnchor(anchor_id)


# ==========================================
# 调度中心：尝试方案 A，失败则回退方案 B
# ==========================================
class HelpManager:
    def __init__(self):
        self.fallback_viewer = None
        self.server = None
        self.port = None

    def show_help(self):
        # 单例模式管理 server，避免每次点击都开启新的端口和线程
        if not self.server:
            try:
                # 直接使用原生的 HTTPServer，不需要自定义 HelpServer 了
                self.server = HTTPServer(('127.0.0.1', 0), HelpRequestHandler)
                self.port = self.server.server_port
                thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                thread.start()
            except Exception as e:
                logger.warning(f"Server start failed: {e}")
                self.open_fallback_gui()
                return

        try:
            url = f"http://127.0.0.1:{self.port}/"
            success = webbrowser.open(url)
            if not success:
                raise RuntimeError("webbrowser.open() returned False")
        except Exception as e:
            logger.warning(f"Browser mode failed: {e}. Falling back to basic GUI mode.")
            self.open_fallback_gui()

    def open_fallback_gui(self):
        # 动态获取当前语言
        lang = SingletonConfig().config.get('language', 'en')
        md_file = 'helpZH.md' if lang == 'zh' else 'help.md'
        md_file_path = get_resource_path(md_file)

        # 如果窗口尚未创建或已被关闭
        if self.fallback_viewer is None or not self.fallback_viewer.isVisible():
            self.fallback_viewer = FallbackMDViewer(md_file_path)
            self.fallback_viewer.show()
        else:
            # 如果窗口已经在显示，直接更新绑定的文件并重新加载
            self.fallback_viewer.md_file = md_file_path
            self.fallback_viewer.loadMarkdown()
            self.fallback_viewer.raise_()
            self.fallback_viewer.activateWindow()


if __name__ == '__main__':
    app = QApplication.instance() or QApplication(sys.argv)

    manager = HelpManager()

    manager.show_help()
    # manager.open_fallback_gui()

    sys.exit(app.exec_())
