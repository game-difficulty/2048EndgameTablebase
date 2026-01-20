import sys
import re
import markdown
import html

from PyQt5 import QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QListWidget, QVBoxLayout, QWidget, QSplitter, QHBoxLayout, \
    QLineEdit, QPushButton, QShortcut, QFrame, QLabel, QCheckBox
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import Qt

from Config import ColorManager, SingletonConfig


# noinspection PyAttributeOutsideInit
class MDViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        lang = SingletonConfig().config.get('language', 'en')
        self.MD_file = 'helpZH.md' if lang == 'zh' else 'help.md'
        self.toc_anchors = []
        self.setupUI()
        self.setupSearchShortcuts()
        self.browser.page().findTextFinished.connect(self.handleFindResult)

    def setupUI(self):
        self.setWindowIcon(QIcon(r"pic\2048_2.ico"))
        self.setWindowTitle(self.tr("Help"))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout()

        # --- 增强型搜索栏 ---
        self.search_bar = QFrame()  # 改用 QFrame 方便设置边框
        self.search_bar.setFixedHeight(45)  # 限制搜索栏高度
        self.search_layout = QHBoxLayout(self.search_bar)
        self.search_layout.setContentsMargins(10, 0, 10, 0)

        # 1. 输入框
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(self.tr("Search..."))
        self.search_input.textChanged.connect(self.updateSearch)  # type: ignore
        self.search_input.returnPressed.connect(self.searchNext)  # type: ignore

        # 2. 结果计数标签
        self.count_label = QLabel("0/0")
        self.count_label.setStyleSheet("color: gray; margin-right: 5px;")

        # 3. 区分大小写勾选框
        self.case_check = QCheckBox("Cc")
        self.case_check.setStyleSheet("font-weight: bold;")
        self.case_check.stateChanged.connect(lambda: self.updateSearch(is_refresh=True))  # type: ignore

        # 4. 导航按钮
        self.btn_prev = QPushButton(" < ")
        self.btn_next = QPushButton(" > ")
        self.btn_prev.clicked.connect(self.searchPrev)  # type: ignore
        self.btn_next.clicked.connect(self.searchNext)  # type: ignore

        # 5. 关闭按钮 (X)
        self.btn_close = QPushButton(" × ")
        self.btn_close.setFixedSize(25, 25)
        self.btn_close.setStyleSheet("font-weight: bold; border: none;")
        self.btn_close.clicked.connect(self.hideSearchBar)  # type: ignore

        # 组装布局
        self.search_layout.addWidget(self.search_input)
        self.search_layout.addWidget(self.count_label)
        self.search_layout.addWidget(self.case_check)
        self.search_layout.addWidget(self.btn_prev)
        self.search_layout.addWidget(self.btn_next)
        self.search_layout.addWidget(self.btn_close)

        self.search_bar.hide()
        self.layout.addWidget(self.search_bar)

        # 统一使用 ColorManager 设置样式
        self.applySearchStyle()

        # ... 原有 Splitter 处理代码 ...
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.toc_list = QListWidget()
        self.toc_list.currentRowChanged.connect(self.onTOCClicked)  # type: ignore
        self.browser = QWebEngineView()
        splitter.addWidget(self.toc_list)
        splitter.addWidget(self.browser)
        self.layout.addWidget(splitter)
        main_widget.setLayout(self.layout)

        self.loadMarkdown()
        self.resize(1320, 800)

    def applySearchStyle(self):
        color_mgr = ColorManager()
        style = f"""
            QFrame {{ background-color: {color_mgr.get_css_color(3)}; border-bottom: 1px solid #d1d5da; }}
            QLineEdit {{ background-color: {color_mgr.get_css_color(6)}; color: {color_mgr.get_css_color(10)}; border-radius: 4px; padding: 2px; }}
            QPushButton:hover {{ background-color: rgba(0,0,0,0.1); }}
            QLabel {{}}
        """
        self.search_bar.setStyleSheet(style)

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
            # 1. 移除可能嵌套的 HTML 标签
            clean_text = re.sub(r'<.*?>', '', text).strip()
            # 2. 关键修复：反转义 HTML 字符（将 &amp; 还原为 &）
            final_text = html.unescape(clean_text)

            self.toc_anchors.append(anchor_id)
            # 根据标题级别添加缩进
            self.toc_list.addItem('    ' * (int(level) - 1) + final_text)

    def onTOCClicked(self, index):
        if 0 <= index < len(self.toc_anchors):
            anchor_id = self.toc_anchors[index]
            # 3. 使用 JavaScript 实现页面内跳转
            js_code = f"document.getElementById('{anchor_id}').scrollIntoView();"
            self.browser.page().runJavaScript(js_code)

    def setupSearchShortcuts(self):
        """绑定快捷键"""
        # Ctrl+F: 显示/聚焦搜索框
        self.shortcut_f = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_f.activated.connect(self.showSearchBar)  # type: ignore

        # Esc: 隐藏搜索框并清除高亮
        self.shortcut_esc = QShortcut(QKeySequence("Esc"), self)
        self.shortcut_esc.activated.connect(self.hideSearchBar)  # type: ignore

    def updateSearch(self, is_refresh=False):
        """
        当文字或选项改变时重新搜索。
        is_refresh 为 True 时（如点击 Cc），重置搜索起点防止“下跳”。
        """
        if is_refresh:
            # 核心修复：通过传入空字符串，强制 WebEngine 清除当前选中和搜索状态
            # 这样接下来的搜索就会从文档顶部重新开始，并应用新的 Cc 标志
            self.browser.page().findText("")

        self.searchText(self.search_input.text())

    def searchText(self, text, forward=True):
        """
        仅执行搜索指令，不在此处处理结果。
        结果将由 findTextFinished 信号异步通知。
        """
        if not text:
            self.count_label.setText("0/0")
            self.browser.page().findText("")  # 清除高亮
            return

        options = QWebEnginePage.FindFlags()
        if not forward:
            options |= QWebEnginePage.FindFlag.FindBackward
        if self.case_check.isChecked():
            options |= QWebEnginePage.FindFlag.FindCaseSensitively

        # 纯粹的指令发起，不再传入回调函数
        self.browser.page().findText(text, options)

    def handleFindResult(self, result):
        """
        结果处理槽函数。
        """
        # 此时 result 是一个包含 activeMatchIndex 和 numberOfMatches 的对象
        total = result.numberOfMatches()
        current = result.activeMatch() + 1 if total > 0 else 0

        self.count_label.setText(f"{current}/{total}")

        # 交互优化：如果没有匹配项，输入框背景可以微调（变红/变黄）
        if total == 0 and self.search_input.text():
            self.count_label.setStyleSheet("color: #f85149; font-weight: bold;")
        else:
            self.count_label.setStyleSheet("")
            self.applySearchStyle()  # 恢复原有样式

    def searchNext(self):
        self.searchText(self.search_input.text(), forward=True)

    def searchPrev(self):
        self.searchText(self.search_input.text(), forward=False)

    def hideSearchBar(self):
        self.search_bar.hide()
        self.browser.findText("")  # 清除高亮
        self.browser.setFocus()

    def showSearchBar(self):
        self.search_bar.show()
        self.search_input.setFocus()
        self.search_input.selectAll()

    def retranslateUi(self):
        """核心修复：实现重新翻译逻辑"""
        # 1. 更新窗口标题
        self.setWindowTitle(self.tr("Help"))

        # 2. 根据最新语言配置重新选择文件路径
        lang = SingletonConfig().config.get('language', 'en')
        self.MD_file = 'helpZH.md' if lang == 'zh' else 'help.md'

        # 3. 重新读取文件并渲染 HTML
        self.loadMarkdown()

        # 4. 如果有搜索结果，在切换语言后清除，防止定位错乱
        self.search_input.clear()


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)

    viewer = MDViewer()
    viewer.show()
    sys.exit(app.exec_())
