import sys
from PyQt5.QtWidgets import QMainWindow, QTextBrowser, QApplication, QListWidget, QVBoxLayout, QWidget, QSplitter
from PyQt5.QtGui import QIcon, QTextCursor
from PyQt5.QtCore import Qt
import re
import markdown


# noinspection PyAttributeOutsideInit
class MDViewer(QMainWindow):
    def __init__(self, MD_file):
        super().__init__()
        self.MD_file = MD_file
        self.toc = []  # Table of contents list
        self.setupUI()

    def setupUI(self):
        self.setWindowIcon(QIcon(r"pic\2048_2.ico"))
        self.setWindowTitle("Markdown Viewer")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.text_browser = QTextBrowser()
        self.toc_list = QListWidget()
        self.toc_list.currentRowChanged.connect(self.onTOCClicked)  # type: ignore

        splitter.addWidget(self.toc_list)
        splitter.addWidget(self.text_browser)
        splitter.setSizes([200, 600])

        layout.addWidget(splitter)
        main_widget.setLayout(layout)

        self.loadMarkdown()
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                font-family: 'Times New Roman';  /* Setting font */
                font-size: 24px;  /* Setting font size */
                line-height: 1.5;  /* Setting line spacing */
                color: #333;  /* Setting font color */
                background-color: #f8f8f8;  /* Setting background color */
                padding: 10px;  /* Setting padding */
            }
        """)

        self.resize(1600, 800)

    def loadMarkdown(self):
        with open(self.MD_file, 'r', encoding='utf-8') as file:
            MD_content = file.read()

        html_text = markdown.markdown(MD_content, extensions=['tables'])
        # 为表格添加样式
        html_text = html_text.replace('<table>', '<table style="border-collapse: collapse; width: 100%;">')
        html_text = html_text.replace('<tr>', '<tr style="border: 1px solid black;">')
        html_text = html_text.replace('<td>', '<td style="border: 1px solid black; padding: 5px;">')
        html_text = html_text.replace('<th>', '<th style="border: 1px solid black; padding: 5px; text-align: center;">')

        self.text_browser.setHtml(html_text)
        self.extractHeaders(html_text)

    def extractHeaders(self, html_text):
        # 定义一个正则表达式，匹配所有的 <h1> 到 <h6> 标签
        header_pattern = re.compile(r'<h([1-6])>(.*?)</h\1>', re.DOTALL)

        self.toc.clear()  # 清空现有的目录
        self.toc_list.clear()  # 清空目录列表

        # 查找所有的标题
        headers = header_pattern.findall(html_text)

        # 遍历所有标题
        for level, header_text in headers:
            # 由于标题中的文本可能有多余的空白字符，使用 strip() 去除
            header_text = header_text.strip()

            # 获取文本光标
            cursor = QTextCursor(self.text_browser.document())
            line_number = 0
            # 遍历文档查找该标题的具体位置
            while not cursor.atEnd():
                cursor.movePosition(QTextCursor.StartOfBlock)  # 移动到当前块的起始位置
                text = cursor.block().text().strip()  # 获取当前块的文本

                # 判断是否找到匹配的标题
                if text == header_text:
                    line_number = cursor.blockNumber() + 1  # 获取标题的行号（从1开始）
                    break
                cursor.movePosition(QTextCursor.NextBlock)  # 向下移动到下一行

            if line_number:
                # 将标题及其行号存入目录
                self.toc.append((header_text, line_number))
                self.toc_list.addItem('  ' * (int(level) - 1) + header_text)  # 添加到目录列表显示

    def onTOCClicked(self, index):
        header, line_number = self.toc[index]  # 获取对应的标题及行号
        cursor = self.text_browser.textCursor()  # 获取当前 QTextCursor
        # 将光标移动到文档末尾
        cursor.movePosition(QTextCursor.End)
        self.text_browser.setTextCursor(cursor)  # 更新 QTextBrowser 的光标位置

        document = self.text_browser.document()  # 获取 QTextDocument
        # 使用 findBlockByLineNumber() 定位到指定行
        block = document.findBlockByLineNumber(line_number - 1)  # line_number 是从 1 开始，blockNumber 从 0 开始
        if block.isValid():  # 如果找到了有效的文本块
            cursor.setPosition(block.position())  # 设置光标位置为该块的起始位置
            self.text_browser.setTextCursor(cursor)  # 更新 QTextBrowser 的光标位置
            self.text_browser.ensureCursorVisible()  # 确保光标可见


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MDViewer('help.md')
    viewer.show()
    sys.exit(app.exec_())
