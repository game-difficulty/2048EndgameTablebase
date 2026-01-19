import sys
import re
import markdown
from PyQt5.QtWidgets import QMainWindow, QApplication, QListWidget, QVBoxLayout, QWidget, QSplitter
from PyQt5.QtGui import QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt

from Config import ColorManager


class MDViewer(QMainWindow):
    def __init__(self, md_file):
        super().__init__()
        self.MD_file = md_file
        self.toc_anchors = []  # 存储 anchor ID
        self.setupUI()

    def setupUI(self):
        self.setWindowIcon(QIcon(r"pic\2048_2.ico"))
        self.setWindowTitle(self.tr("Help - Professional Edition"))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. 使用 QWebEngineView 替换 QTextBrowser
        self.browser = QWebEngineView()

        self.toc_list = QListWidget()
        self.toc_list.currentRowChanged.connect(self.onTOCClicked)

        splitter.addWidget(self.toc_list)
        splitter.addWidget(self.browser)
        splitter.setSizes([300, 1300])

        layout.addWidget(splitter)
        main_widget.setLayout(layout)

        self.loadMarkdown()
        self.resize(1600, 800)

    def loadMarkdown(self):
        with open(self.MD_file, 'r', encoding='utf-8') as file:
            md_content = file.read()

        # 使用这些扩展来模拟 GitHub 的行为
        extensions = [
            'extra',  # 包含 tables, fenced_code, attr_list 等
            'sane_lists',  # 让列表解析更符合直觉，防止不同类型的列表混淆
            'nl2br',  # 换行符转 <br>
            'toc'  # 生成锚点
        ]

        html_body = markdown.markdown(md_content, extensions=extensions)

        # 2. 构建完整的 HTML 骨架，注入现代样式和公式引擎
        color_mgr = ColorManager()
        bg_color = color_mgr.get_css_color(1)
        text_color = color_mgr.get_css_color(10)

        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script>
            window.MathJax = {{
                tex: {{
                    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                    processEscapes: true,
                    processEnvironments: true
                }},
                options: {{
                    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
                }},
                startup: {{
                    pageReady: () => {{
                        return MathJax.startup.defaultPageReady().then(() => {{
                            console.log('MathJax is ready!');
                        }});
                    }}
                }}
            }};
            </script>
            
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

            <style>
                /* 1. 基础页面设置 */
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
                    font-size: 16px;
                    line-height: 1.6;
                    word-wrap: break-word;
                    color: {text_color};
                    background-color: {bg_color};
                    padding: 45px;
                    max-width: 1000px;
                    margin: 0 auto;
                    scroll-behavior: smooth;
                }}

                /* 2. 修复列表渲染问题 */
                .markdown-body ul {{
                    list-style-type: disc !important; /* 强制无序列表显示圆点 */
                    padding-left: 2em !important;
                    margin-top: 0;
                    margin-bottom: 16px;
                }}

                .markdown-body ol {{
                    list-style-type: decimal !important; /* 强制有序列表显示数字 */
                    padding-left: 2em !important;
                    margin-top: 0;
                    margin-bottom: 16px;
                }}

                .markdown-body li {{
                    display: list-item !important;
                    margin-top: 0.25em;
                }}

                /* 嵌套列表样式：确保层级清晰 */
                .markdown-body li > ul {{
                    list-style-type: circle !important; /* 二级列表圆圈 */
                    margin-top: 0;
                    margin-bottom: 0;
                }}

                .markdown-body li > ol {{
                    margin-top: 0;
                    margin-bottom: 0;
                }}

                /* 3. 标题样式 */
                .markdown-body h1, .markdown-body h2 {{
                    border-bottom: 1px solid #eaecef;
                    padding-bottom: 0.3em;
                    margin-top: 24px;
                    margin-bottom: 16px;
                    font-weight: 600;
                }}

                /* 4. 表格样式 (GitHub 风格) */
                .markdown-body table {{
                    display: table;
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 0;
                    margin-bottom: 16px;
                }}
                .markdown-body table th, .markdown-body table td {{
                    padding: 6px 13px;
                    border: 1px solid #dfe2e5;
                }}
                .markdown-body table tr {{
                    background-color: {bg_color};
                    border-top: 1px solid #c6cbd1;
                }}
                .markdown-body table tr:nth-child(even) {{
                    background-color: rgba(0,0,0,0.03);
                }}

                /* 5. 代码块样式 */
                .markdown-body pre {{
                    background-color: rgba(0,0,0,0.05);
                    border-radius: 6px;
                    padding: 16px;
                    overflow: auto;
                    font-size: 85%;
                    line-height: 1.45;
                }}
                .markdown-body code {{
                    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                    padding: 0.2em 0.4em;
                    margin: 0;
                    font-size: 85%;
                    background-color: rgba(27,31,35,0.05);
                    border-radius: 3px;
                }}
                .markdown-body pre code {{
                    background-color: transparent;
                    padding: 0;
                }}

                /* 6. 滚动条美化 */
                ::-webkit-scrollbar {{ width: 10px; }}
                ::-webkit-scrollbar-track {{ background: transparent; }}
                ::-webkit-scrollbar-thumb {{ background: #d1d5da; border-radius: 5px; }}
                ::-webkit-scrollbar-thumb:hover {{ background: #adb5bd; }}
            </style>
        </head>
        <body class="markdown-body">
            {html_body}
        </body>
        </html>
        """

        self.browser.setHtml(full_html)
        self.extractHeaders(html_body)

    def extractHeaders(self, html_body):
        # 匹配 markdown 生成的带 id 的标题
        pattern = re.compile(r'<h([1-6]) id="(.*?)">(.*?)</h\1>')
        headers = pattern.findall(html_body)

        self.toc_anchors.clear()
        self.toc_list.clear()

        for level, anchor_id, text in headers:
            # 清理标题内可能存在的 HTML 标签
            clean_text = re.sub(r'<.*?>', '', text).strip()
            self.toc_anchors.append(anchor_id)
            # 根据标题级别添加缩进
            self.toc_list.addItem('    ' * (int(level) - 1) + clean_text)

    def onTOCClicked(self, index):
        if 0 <= index < len(self.toc_anchors):
            anchor_id = self.toc_anchors[index]
            # 3. 使用 JavaScript 实现页面内跳转
            js_code = f"document.getElementById('{anchor_id}').scrollIntoView();"
            self.browser.page().runJavaScript(js_code)


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)

    viewer = MDViewer('help.md')
    viewer.show()
    sys.exit(app.exec_())
