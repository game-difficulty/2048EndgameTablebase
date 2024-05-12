from PyQt5.QtWidgets import QMainWindow, QTextBrowser
from PyQt5.QtGui import QIcon


# noinspection PyAttributeOutsideInit
class HTMLViewer(QMainWindow):
    def __init__(self, html_file):
        super().__init__()
        self.html_file = html_file
        self.setupUI()

    def setupUI(self):
        self.setWindowIcon(QIcon(r"pic\2048_2.ico"))
        self.text_browser = QTextBrowser()
        self.setCentralWidget(self.text_browser)
        with open(self.html_file, 'r') as file:
            html_content = file.read()
        self.text_browser.setHtml(html_content)
        self.setWindowTitle("Help")
        self.resize(800, 600)
