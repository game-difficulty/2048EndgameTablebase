import sys
from PyQt5.QtWidgets import QMainWindow, QTextBrowser, QApplication, QListWidget, QVBoxLayout, QWidget, QSplitter
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import re


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

        self.toc_list.currentRowChanged.connect(self.onTOCClicked)  # type: ignore
        self.resize(1000, 600)

    def loadMarkdown(self):
        with open(self.MD_file, 'r', encoding='utf-8') as file:
            MD_content = file.read()

        self.text_browser.setMarkdown(MD_content)
        self.extractHeaders(MD_content)

    def extractHeaders(self, content):
        pattern = re.compile(r'^(#{1,6})\s*(.*)', re.MULTILINE)
        for match in pattern.finditer(content):
            level = len(match.group(1))
            header = match.group(2).strip()
            pos = match.start()
            self.toc.append((header, pos))
            self.toc_list.addItem('  ' * (level - 1) + header)

    def onTOCClicked(self, index):
        header, pos = self.toc[index]
        cursor = self.text_browser.textCursor()  # 获取当前光标
        cursor.setPosition(pos)  # 设置光标位置到目标位置
        self.text_browser.setTextCursor(cursor)  # 更新 QTextBrowser 的光标位置
        self.text_browser.ensureCursorVisible()  # 确保光标可见


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = MDViewer('help.md')
    viewer.show()
    sys.exit(app.exec_())
