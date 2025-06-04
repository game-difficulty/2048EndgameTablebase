import os
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

from BookBuilder import start_build
from Config import SingletonConfig, formation_info


# noinspection PyAttributeOutsideInit
class SettingsWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        config = SingletonConfig().config
        self.setObjectName("self")
        self.setWindowIcon(QtGui.QIcon(r"pic\settings.ico"))
        self.resize(840, 600)
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
        self.pattern_text.setObjectName("pattern_text")
        self.pattern_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.pattern_text, 0, 0, 1, 1)
        self.filepath_text = QtWidgets.QLabel(self.centralwidget)
        self.filepath_text.setObjectName("filepath_text")
        self.filepath_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.filepath_text, 1, 0, 1, 1)
        self.filepath_edit = QtWidgets.QTextEdit(self.centralwidget)
        self.filepath_edit.setObjectName("filepath_edit")
        self.filepath_edit.setMaximumSize(QtCore.QSize(600, 80))
        self.filepath_edit.setMinimumSize(QtCore.QSize(420, 60))
        self.selfLayout.addWidget(self.filepath_edit, 1, 1, 1, 1)
        self.set_filepath_bt = QtWidgets.QPushButton(self.centralwidget)
        self.set_filepath_bt.setObjectName("set_filepath_bt")
        self.set_filepath_bt.setMinimumSize(QtCore.QSize(180, 20))
        self.set_filepath_bt.clicked.connect(self.filepath_changed)  # type: ignore
        self.selfLayout.addWidget(self.set_filepath_bt, 1, 2, 1, 1)
        self.target_combo = QtWidgets.QComboBox(self.centralwidget)
        self.target_combo.setObjectName("target_combo")
        for i in ["256", "512", "1024", "2048", "4096"]:
            self.target_combo.addItem(i)
        self.selfLayout.addWidget(self.target_combo, 0, 2, 1, 1)
        self.pattern_combo = QtWidgets.QComboBox(self.centralwidget)
        self.pattern_combo.setObjectName("pattern_combo")
        for i in formation_info.keys():
            self.pattern_combo.addItem(i)
        self.selfLayout.addWidget(self.pattern_combo, 0, 1, 1, 1)
        self.build_bt = QtWidgets.QPushButton(self.centralwidget)
        self.build_bt.setObjectName("build_bt")
        self.build_bt.setMinimumSize(QtCore.QSize(180, 20))
        self.build_bt.clicked.connect(self.build_book)  # type: ignore
        self.selfLayout.addWidget(self.build_bt, 1, 3, 1, 1)

        self.steps_label = QtWidgets.QLabel(self.centralwidget)
        self.steps_label.setObjectName("steps_label")
        self.steps_label.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.steps_label, 2, 0, 1, 1)

        # 步数输入框
        self.steps_input = QtWidgets.QSpinBox(self.centralwidget)
        self.steps_input.setObjectName("steps_input")
        self.steps_input.setRange(0, 10000)
        self.steps_input.setSingleStep(100)
        self.selfLayout.addWidget(self.steps_input, 2, 1, 1, 1)

        self.hline = QtWidgets.QFrame(self.centralwidget)
        self.hline.setFrameShape(QtWidgets.QFrame.HLine)
        self.hline.setStyleSheet("border-width: 3px; border-style: inset;")
        self.hline.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.selfLayout.addWidget(self.hline, 5, 0, 1, 4)

        self.colorset_text = QtWidgets.QLabel(self.centralwidget)
        self.colorset_text.setObjectName("colorset_text")
        self.colorset_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.colorset_text, 6, 0, 1, 1)
        self.color_combo = QtWidgets.QComboBox(self.centralwidget)
        self.color_combo.setObjectName("color_combo")
        for i in range(1, 16):
            self.color_combo.addItem(str(2 ** i))
        self.selfLayout.addWidget(self.color_combo, 6, 1, 1, 1)
        self.color_bt = QtWidgets.QPushButton(self.centralwidget)
        self.color_bt.clicked.connect(self.show_ColorDialog)  # type: ignore
        self.color_bt.setObjectName("color_bt")
        self.selfLayout.addWidget(self.color_bt, 6, 2, 1, 1)
        self.color_default_bt = QtWidgets.QPushButton(self.centralwidget)
        self.color_default_bt.clicked.connect(self.set_default_color)  # type: ignore
        self.color_default_bt.setObjectName("color_default_bt")
        self.selfLayout.addWidget(self.color_default_bt, 6, 3, 1, 1)

        self.spawnrate_text = QtWidgets.QLabel(self.centralwidget)
        self.spawnrate_text.setObjectName("spawnrate_text")
        self.spawnrate_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.spawnrate_text, 7, 0, 1, 1)
        self.spawnrate_box = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.spawnrate_box.setObjectName("spawnrate_box")
        self.spawnrate_box.setRange(0.0, 1.0)
        self.spawnrate_box.setSingleStep(0.01)
        self.spawnrate_box.setValue(SingletonConfig().config['4_spawn_rate'])
        self.spawnrate_box.valueChanged.connect(self.update_spawn_rate)  # type: ignore
        self.selfLayout.addWidget(self.spawnrate_box, 7, 1, 1, 1)
        self.infoButton = QtWidgets.QPushButton()
        self.infoButton.setIcon(QtGui.QIcon(r'pic\OQM.png'))
        self.infoButton.setIconSize(QtCore.QSize(24, 24))
        self.infoButton.setFlat(True)
        self.selfLayout.addWidget(self.infoButton, 7, 2, 1, 1)
        self.infoButton.clicked.connect(self.show_message)  # type: ignore

        self.demo_speed_text = QtWidgets.QLabel(self.centralwidget)
        self.demo_speed_text.setObjectName("demo_speed_text")
        self.demo_speed_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.demo_speed_text, 8, 0, 1, 1)
        self.demo_speed_box = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.centralwidget)
        self.demo_speed_box.setMinimum(1)
        self.demo_speed_box.setMaximum(200)
        self.demo_speed_box.setValue(config.get('demo_speed', 10))
        self.demo_speed_box.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.demo_speed_box.setTickInterval(10)
        self.demo_speed_box.valueChanged.connect(self.demo_speed_changed)  # type: ignore
        self.demo_speed_box.setObjectName("demo_speed_box")
        self.selfLayout.addWidget(self.demo_speed_box, 8, 1, 1, 3)

        self.anim_text = QtWidgets.QLabel(self.centralwidget)
        self.anim_text.setObjectName("anim_text")
        self.anim_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.anim_text, 9, 0, 1, 1)
        self.appear_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.appear_checkBox.setObjectName("appear_checkBox")
        if config.get('do_animation', (False, False))[0]:
            self.appear_checkBox.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.appear_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.selfLayout.addWidget(self.appear_checkBox, 9, 1, 1, 1)
        self.pop_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.pop_checkBox.setObjectName("pop_checkBox")
        if config.get('do_animation', (False, False))[1]:
            self.pop_checkBox.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.pop_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.selfLayout.addWidget(self.pop_checkBox, 9, 2, 1, 1)

        self.tile_font_size_text = QtWidgets.QLabel(self.centralwidget)
        self.tile_font_size_text.setObjectName("tile_font_size_text")
        self.tile_font_size_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.tile_font_size_text, 10, 0, 1, 1)
        self.tile_font_size_box = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self.centralwidget)
        self.tile_font_size_box.setMinimum(50)
        self.tile_font_size_box.setMaximum(150)
        self.tile_font_size_box.setValue(config.get('font_size_factor', 100))
        self.tile_font_size_box.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.tile_font_size_box.setTickInterval(5)
        self.tile_font_size_box.valueChanged.connect(self.font_size_changed)  # type: ignore
        self.tile_font_size_box.setObjectName("tile_font_size_box")
        self.selfLayout.addWidget(self.tile_font_size_box, 10, 1, 1, 3)

        self.PageLayout.addLayout(self.selfLayout)
        self.save_bt = QtWidgets.QPushButton(self.centralwidget)
        self.save_bt.setObjectName("save_bt")
        self.save_bt.clicked.connect(self.save_all)  # type: ignore
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

        self.steps_label.setText(_translate("Settings", "Steps:"))
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
        spawn_rate = SingletonConfig().config['4_spawn_rate']
        if spawn_rate != 0.1:
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Warning",
                f"The current 4_spawn_rate is {spawn_rate}, which is not the default 0.1. Do you want to continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.No:
                return

        pattern = self.pattern_combo.currentText()
        target = self.target_combo.currentText()
        pathname = self.filepath_edit.toPlainText()
        if pattern and target and pathname and os.path.exists(pathname):
            config = SingletonConfig().config
            if pattern in ['444', 'LL', 'L3']:
                ptn = pattern + '_' + target
                config['filepath_map'][ptn] = [pathname, ]
                pathname = os.path.join(pathname, ptn + '_')
            else:
                ptn = pattern + '_' + target
                config['filepath_map'][ptn] = [pathname, ]
                pathname = os.path.join(pathname, ptn + '_')
            target = int(np.log2(int(target)))
            self.build_bt.setText('Building...')
            self.build_bt.setEnabled(False)
            self.Building_thread = BuildThread(pattern, target, pathname, self.steps_input.value())
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
        if self.advanced_algo_checkBox.isChecked() and self.compress_checkBox.isChecked():
            self.compress_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        SingletonConfig().config['compress'] = self.compress_checkBox.isChecked()

    def compress_temp_files_state_changed(self):
        SingletonConfig().config['compress_temp_files'] = self.compress_temp_files_checkBox.isChecked()

    def update_spawn_rate(self):
        SingletonConfig().config['4_spawn_rate'] = float(self.spawnrate_box.text().replace(',', '.'))

    def show_message(self):
        QtWidgets.QMessageBox.information(self, 'Information', 'This affects all modules of the program, ' +
                                          '''make sure you know what you're doing''')

    def save_all(self):

        SingletonConfig().config['4_spawn_rate'] = float(self.spawnrate_box.text().replace(',', '.'))
        SingletonConfig().config['do_animation'] = [self.appear_checkBox.isChecked(), self.pop_checkBox.isChecked()]

        SingletonConfig.save_config(SingletonConfig().config)


class BuildThread(QtCore.QThread):
    finished = QtCore.pyqtSignal()

    def __init__(self, pattern, target, pathname, steps):
        super().__init__()
        self.pattern = pattern
        self.target = target
        self.pathname = pathname
        self.steps = steps

    def run(self):
        start_build(self.pattern, self.target, self.pathname, self.steps)
        self.finished.emit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = SettingsWindow()
    main.show()
    sys.exit(app.exec_())
