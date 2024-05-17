import os
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

from BookBuilder import start_build
from Config import SingletonConfig


# noinspection PyAttributeOutsideInit
class SettingsWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        config = SingletonConfig().config
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\settings.ico"))
        self.resize(840, 480)
        self.setStyleSheet("QMainWindow{\n"
                           "    background-color: rgb(245, 245, 247);\n"
                           "}")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.PageLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.PageLayout.setObjectName("PageLayout")
        self.selfLayout = QtWidgets.QGridLayout()
        self.selfLayout.setContentsMargins(25, 25, 25, 25)
        self.selfLayout.setHorizontalSpacing(15)
        self.selfLayout.setVerticalSpacing(40)
        self.selfLayout.setObjectName("selfLayout")

        self.pattern_text = QtWidgets.QLabel(self.centralwidget)
        self.pattern_text.setObjectName("colorset_text")
        self.pattern_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.pattern_text, 0, 0, 1, 1)
        self.filepath_text = QtWidgets.QLabel(self.centralwidget)
        self.filepath_text.setObjectName("colorset_text")
        self.filepath_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.filepath_text, 1, 0, 1, 1)
        self.filepath_edit = QtWidgets.QTextEdit(self.centralwidget)
        self.filepath_edit.setObjectName("filepath_edit")
        self.filepath_edit.setMaximumSize(QtCore.QSize(600, 64))
        self.selfLayout.addWidget(self.filepath_edit, 1, 1, 1, 1)
        self.set_filepath_bt = QtWidgets.QPushButton(self.centralwidget)
        self.set_filepath_bt.setObjectName("set_filepath_bt")
        self.set_filepath_bt.clicked.connect(self.filepath_changed)
        self.selfLayout.addWidget(self.set_filepath_bt, 1, 2, 1, 1)
        self.target_combo = QtWidgets.QComboBox(self.centralwidget)
        self.target_combo.setObjectName("target_combo")
        for i in ["128", "256", "512", "1024", "2048", "4096", "8192"]:
            self.target_combo.addItem(i)
        self.selfLayout.addWidget(self.target_combo, 0, 2, 1, 1)
        self.pattern_combo = QtWidgets.QComboBox(self.centralwidget)
        self.pattern_combo.setObjectName("pattern_combo")
        for i in ["444", "4431", "LL", "L3", "t", "442", "free8", "free9", "free10", "4441", "4432", "free8w", "free9w",
                  "free10w", "free11w"]:
            self.pattern_combo.addItem(i)
        self.selfLayout.addWidget(self.pattern_combo, 0, 1, 1, 1)
        self.pos_combo = QtWidgets.QComboBox(self.centralwidget)
        self.pos_combo.setObjectName("target_combo")
        for i in ["0", "1", "2"]:
            self.pos_combo.addItem(i)
        self.selfLayout.addWidget(self.pos_combo, 0, 3, 1, 1)
        self.build_bt = QtWidgets.QPushButton(self.centralwidget)
        self.build_bt.setObjectName("build_bt")
        self.build_bt.clicked.connect(self.build_book)
        self.selfLayout.addWidget(self.build_bt, 1, 3, 1, 1)

        self.options_text = QtWidgets.QLabel(self.centralwidget)
        self.options_text.setObjectName("options_text")
        self.options_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.options_text, 2, 0, 1, 1)
        self.compress_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.compress_checkBox.setObjectName("compress_checkBox")
        self.selfLayout.addWidget(self.compress_checkBox, 2, 1, 1, 1)
        if config.get('compress', False):
            self.compress_checkBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.compress_checkBox.setCheckState(QtCore.Qt.Unchecked)
        self.compress_checkBox.stateChanged.connect(self.compress_state_changed)

        self.hline = QtWidgets.QFrame(self.centralwidget)
        self.hline.setFrameShape(QtWidgets.QFrame.HLine)
        self.hline.setStyleSheet("border-width: 3px; border-style: inset;")
        self.hline.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.selfLayout.addWidget(self.hline, 4, 0, 1, 4)

        self.colorset_text = QtWidgets.QLabel(self.centralwidget)
        self.colorset_text.setObjectName("colorset_text")
        self.colorset_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.colorset_text, 5, 0, 1, 1)
        self.color_combo = QtWidgets.QComboBox(self.centralwidget)
        self.color_combo.setObjectName("color_combo")
        for i in range(1, 16):
            self.color_combo.addItem(str(2 ** i))
        self.selfLayout.addWidget(self.color_combo, 5, 1, 1, 1)
        self.color_bt = QtWidgets.QPushButton(self.centralwidget)
        self.color_bt.clicked.connect(self.show_ColorDialog)
        self.color_bt.setObjectName("color_bt")
        self.selfLayout.addWidget(self.color_bt, 5, 2, 1, 1)
        self.color_default_bt = QtWidgets.QPushButton(self.centralwidget)
        self.color_default_bt.clicked.connect(self.set_default_color)
        self.color_default_bt.setObjectName("color_default_bt")
        self.selfLayout.addWidget(self.color_default_bt, 5, 3, 1, 1)

        self.spawnrate_text = QtWidgets.QLabel(self.centralwidget)
        self.spawnrate_text.setObjectName("spawnrate_text")
        self.spawnrate_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.spawnrate_text, 6, 0, 1, 1)
        self.spawnrate_box = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.spawnrate_box.setObjectName("spawnrate_box")
        self.selfLayout.addWidget(self.spawnrate_box, 6, 1, 1, 1)
        self.spawnrate_warning = QtWidgets.QLabel(self.centralwidget)
        self.spawnrate_warning.setObjectName("spawnrate_text")
        self.spawnrate_warning.setText("Not supported yet")
        self.selfLayout.addWidget(self.spawnrate_warning, 6, 2, 1, 1)

        self.demo_speed_text = QtWidgets.QLabel(self.centralwidget)
        self.demo_speed_text.setObjectName("demo_speed_text")
        self.demo_speed_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.demo_speed_text, 7, 0, 1, 1)
        self.demo_speed_box = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralwidget)
        self.demo_speed_box.setMinimum(0)
        self.demo_speed_box.setMaximum(200)
        self.demo_speed_box.setValue(config.get('demo_speed',10))
        self.demo_speed_box.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.demo_speed_box.setTickInterval(10)
        self.demo_speed_box.valueChanged.connect(self.demo_speed_changed)
        self.demo_speed_box.setObjectName("demo_speed_box")
        self.selfLayout.addWidget(self.demo_speed_box, 7, 1, 1, 3)

        self.anim_text = QtWidgets.QLabel(self.centralwidget)
        self.anim_text.setObjectName("anim_text")
        self.anim_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.anim_text, 8, 0, 1, 1)
        self.appear_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.appear_checkBox.setObjectName("appear_checkBox")
        if config.get('do_animation',(False, False))[0]:
            self.appear_checkBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.appear_checkBox.setCheckState(QtCore.Qt.Unchecked)
        self.selfLayout.addWidget(self.appear_checkBox, 8, 1, 1, 1)
        self.pop_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.pop_checkBox.setObjectName("pop_checkBox")
        if config.get('do_animation',(False, False))[1]:
            self.pop_checkBox.setCheckState(QtCore.Qt.Checked)
        else:
            self.pop_checkBox.setCheckState(QtCore.Qt.Unchecked)
        self.selfLayout.addWidget(self.pop_checkBox, 8, 2, 1, 1)

        self.tile_font_size_text = QtWidgets.QLabel(self.centralwidget)
        self.tile_font_size_text.setObjectName("tile_font_size_text")
        self.tile_font_size_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.tile_font_size_text, 10, 0, 1, 1)
        self.tile_font_size_box = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.centralwidget)
        self.tile_font_size_box.setMinimum(50)
        self.tile_font_size_box.setMaximum(150)
        self.tile_font_size_box.setValue(config.get('font_size_factor', 100))
        self.tile_font_size_box.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.tile_font_size_box.setTickInterval(5)
        self.tile_font_size_box.valueChanged.connect(self.font_size_changed)
        self.tile_font_size_box.setObjectName("demo_speed_box")
        self.selfLayout.addWidget(self.tile_font_size_box, 10, 1, 1, 3)

        self.PageLayout.addLayout(self.selfLayout)
        self.save_bt = QtWidgets.QPushButton(self.centralwidget)
        self.save_bt.setObjectName("save_bt")
        self.save_bt.clicked.connect(self.save_all)
        self.PageLayout.addWidget(self.save_bt)
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)

    # noinspection PyTypeChecker
    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Settings", "Settings"))
        self.pattern_text.setText(_translate("Settings", "Pattern:"))
        self.filepath_text.setText(_translate("Settings", "Save to &\nLoad From:"))
        self.build_bt.setText(_translate("Settings", "BUILD"))
        self.options_text.setText(_translate("Settings", "Options:"))
        self.compress_checkBox.setText(_translate("Settings", "Compress"))
        self.set_filepath_bt.setText(_translate("Settings", "SET..."))
        self.pop_checkBox.setText(_translate("Settings", "Pop"))
        self.appear_checkBox.setText(_translate("Settings", "Appear"))
        self.colorset_text.setText(_translate("Settings", "Tile Color:"))
        self.color_bt.setText(_translate("Settings", "Choose Color"))
        self.color_default_bt.setText(_translate("Settings", "Default"))
        self.spawnrate_text.setText(_translate("Settings", "4 Spawn Rate:"))
        self.demo_speed_text.setText(_translate("Settings", "Demo Speed:"))
        self.tile_font_size_text.setText(_translate("Settings", "Tile Font Size:"))
        self.anim_text.setText(_translate("Settings", "Do Animation:"))
        self.save_bt.setText(_translate("Settings", "SAVE"))

    def show_ColorDialog(self):
        config = SingletonConfig().config
        num = self.color_combo.currentText()
        current_color = config['colors'][int(np.log2(int(num))) - 1] if num != '' else '#FFFFFF'
        color_dialog = QtWidgets.QColorDialog(self)
        color_dialog.setWindowIcon(self.windowIcon())
        color = color_dialog.getColor(QtGui.QColor(current_color))
        if color.isValid() and num != '':
            num = int(np.log2(int(num))) - 1
            config['colors'][num] = color.name()

    def set_default_color(self):
        default_colors = ['#043c24', '#06643d', '#1b955b', '#20c175', '#fc56a0', '#e4317f', '#e900ad', '#bf009c',
                          '#94008a', '#6a0079', '#3f0067', '#00406b', '#006b9a', '#0095c8', '#00c0f7', '#00c0f7'] + [
                             '#ffffff'] * 20
        num = int(np.log2(int(self.color_combo.currentText()))) - 1
        SingletonConfig().config['colors'][num] = default_colors[num]

    def filepath_changed(self):
        options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filepath = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if filepath:
            self.filepath_edit.setText(filepath)

    def build_book(self):
        position = self.pos_combo.currentText()
        pattern = self.pattern_combo.currentText()
        target = self.target_combo.currentText()
        pathname = self.filepath_edit.toPlainText()
        if pattern and target and pathname and position and os.path.exists(pathname):
            config = SingletonConfig().config
            if pattern in ['444', 'LL', 'L3']:
                ptn = pattern + '_' + target + '_' + position
                config['filepath_map'][ptn] = pathname
                pathname = os.path.join(pathname, ptn + '_')
            else:
                ptn = pattern + '_' + target
                config['filepath_map'][ptn] = pathname
                pathname = os.path.join(pathname, ptn + '_')
            target = np.log2(int(target))
            position = int(position)
            self.build_bt.setText('Building...')
            self.build_bt.setEnabled(False)
            self.Building_thread = BuildThread(pattern, target, position, pathname)
            self.Building_thread.finished.connect(self.on_build_finished)
            self.Building_thread.start()  # 启动线程

    def on_build_finished(self):
        self.build_bt.setText('BUILD')
        self.build_bt.setEnabled(True)

    def demo_speed_changed(self):
        SingletonConfig().config['demo_speed'] = self.demo_speed_box.value()

    def font_size_changed(self):
        SingletonConfig().config['font_size_factor'] = self.tile_font_size_box.value()

    def compress_state_changed(self):
        SingletonConfig().config['compress'] = self.compress_checkBox.isChecked()

    def save_all(self):
        SingletonConfig().config['4_spawn_rate'] = float(self.spawnrate_box.text())
        SingletonConfig().config['do_animation'] = [self.appear_checkBox.isChecked(), self.pop_checkBox.isChecked()]
        SingletonConfig().config['compress'] = self.compress_checkBox.isChecked()
        SingletonConfig.save_config(SingletonConfig().config)


class BuildThread(QtCore.QThread):
    finished = QtCore.pyqtSignal()

    def __init__(self, pattern, target, position, pathname):
        super().__init__()
        self.pattern = pattern
        self.target = target
        self.position = position
        self.pathname = pathname

    def run(self):
        start_build(self.pattern, self.target, self.position, self.pathname)
        self.finished.emit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = SettingsWindow()
    main.show()
    sys.exit(app.exec_())
