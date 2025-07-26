import os
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QToolButton, QMenu, QAction

from BookBuilder import start_build
from Config import SingletonConfig, category_info, theme_map


class TwoLevelComboBox(QToolButton):
    currentTextChanged = QtCore.pyqtSignal(str)

    def __init__(self, default_text='', parent=None):
        super().__init__(parent)
        self.setText(default_text)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMinimumSize(150, 27)
        self.setMaximumSize(600, 36)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.category_menu = QMenu(self)
        self.setMenu(self.category_menu)
        self.category_data = {}
        self.currentText = ''

    def add_category(self, category_name, items):
        category_action = QAction(category_name, self)
        submenu = QMenu(category_name, self)

        for item in items:
            item_action = QAction(item, self)
            item_action.triggered.connect(lambda _, x=item: self._on_item_selected(x))  # type: ignore
            submenu.addAction(item_action)

        category_action.setMenu(submenu)
        self.category_menu.addAction(category_action)

    def _on_item_selected(self, item):
        self.setText(item)
        self.currentText = item
        self.currentTextChanged.emit(item)

    def retranslateUi(self, new_default_text):
        if not self.currentText:
            self.setText(new_default_text)


class SingleLevelComboBox(QToolButton):
    currentTextChanged = QtCore.pyqtSignal(str)

    def __init__(self, default_text='', parent=None):
        super().__init__(parent)
        self.setText(default_text)
        self.setPopupMode(QToolButton.InstantPopup)
        self.setMinimumSize(90, 20)
        self.setMaximumSize(600, 30)
        # self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.category_menu = QMenu(self)
        self.setMenu(self.category_menu)
        self.category_data = {}
        self.currentText = ''
        self.items = []

    def add_items(self, items):
        for item in items:
            self.add_item(item)

    def add_item(self, item: str):
        """添加单个菜单项"""
        item_action = QAction(item, self)
        item_action.triggered.connect(lambda _, x=item: self._on_item_selected(x))  # type: ignore
        self.category_menu.addAction(item_action)
        self.items.append(item)

    def remove_item(self, item: str):
        """删除指定菜单项"""
        for action in self.category_menu.actions():
            if action.text() == item:
                self.category_menu.removeAction(action)
                action.deleteLater()  # 释放内存
                self.items.remove(item)
                break

    def _on_item_selected(self, item):
        self.setText(item)
        self.currentText = item
        self.currentTextChanged.emit(item)

    def retranslateUi(self, new_default_text):
        if not self.currentText:
            self.setText(new_default_text)


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
        self.selfLayout.setVerticalSpacing(27)
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

        self.target_combo = SingleLevelComboBox(self.tr("Target Tile"), self.centralwidget)
        self.target_combo.setObjectName("target_combo")
        self.target_combo.add_items(["128", "256", "512", "1024", "2048", "4096", "8192"])
        self.selfLayout.addWidget(self.target_combo, 0, 2, 1, 1)

        self.pattern_combo = TwoLevelComboBox(self.tr("Select Formation"), self.centralwidget)
        self.pattern_combo.setObjectName("pattern_combo")
        for category, items in category_info.items():
            self.pattern_combo.add_category(category, items)
        self.selfLayout.addWidget(self.pattern_combo, 0, 1, 1, 1)
        self.pattern_combo.currentTextChanged.connect(self.update_pos_combo_visibility)

        self.pos_combo = SingleLevelComboBox(self.tr("Target Position"), self.centralwidget)
        self.pos_combo.setObjectName("pos_combo")
        self.pos_combo.add_items(["0", "1", "2"])
        self.selfLayout.addWidget(self.pos_combo, 0, 3, 1, 1)
        self.pos_combo.hide()

        self.build_bt = QtWidgets.QPushButton(self.centralwidget)
        self.build_bt.setObjectName("build_bt")
        self.build_bt.setMinimumSize(QtCore.QSize(180, 20))
        self.build_bt.clicked.connect(self.build_book)  # type: ignore
        self.selfLayout.addWidget(self.build_bt, 1, 3, 1, 1)

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

        # 创建主题选择按钮（下拉菜单）
        self.theme_button = QtWidgets.QToolButton(self.centralwidget)
        self.theme_button.setMinimumSize(150, 20)
        self.theme_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.theme_button.setObjectName("theme_button")
        self.selfLayout.addWidget(self.theme_button, 6, 3, 1, 1)
        self.theme_menu = QtWidgets.QMenu(self.theme_button)
        self.theme_button.setMenu(self.theme_menu)

        # 添加主题选项
        for theme_name in theme_map.keys():
            action = self.theme_menu.addAction(theme_name)
            action.triggered.connect(  # type: ignore
                lambda checked, theme=theme_name: self.apply_theme(theme)
            )

        self.spawnrate_text = QtWidgets.QLabel(self.centralwidget)
        self.spawnrate_text.setObjectName("spawnrate_text")
        self.spawnrate_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.spawnrate_text, 7, 2, 1, 1)
        self.spawnrate_box = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.spawnrate_box.setObjectName("spawnrate_box")
        self.spawnrate_box.setRange(0.0, 1.0)
        self.spawnrate_box.setSingleStep(0.01)
        self.spawnrate_box.setValue(SingletonConfig().config['4_spawn_rate'])
        self.spawnrate_box.valueChanged.connect(self.update_spawn_rate)  # type: ignore
        self.selfLayout.addWidget(self.spawnrate_box, 7, 3, 1, 1)
            
        self.data_points_text = QtWidgets.QLabel(self.centralwidget)
        self.data_points_text.setObjectName("data_points_text")
        self.data_points_text.setStyleSheet("font: 500 12pt \"Cambria\";")
        self.selfLayout.addWidget(self.data_points_text, 7, 0, 1, 1)
        self.data_points_box = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.data_points_box.setObjectName("data_points_box")
        self.data_points_box.setRange(16, 4000000000)
        self.data_points_box.setDecimals(0)
        self.data_points_box.setSingleStep(8)
        self.data_points_box.setValue(SingletonConfig().config['data_points'])
        self.data_points_box.valueChanged.connect(self.update_data_points)  # type: ignore
        self.selfLayout.addWidget(self.data_points_box, 7, 1, 1, 1)

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
        self.anim_checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.anim_checkBox.setObjectName("anim_checkBox")
        if config.get('do_animation', True):
            self.anim_checkBox.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.anim_checkBox.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.selfLayout.addWidget(self.anim_checkBox, 9, 1, 1, 1)
        # 创建语言选择按钮（下拉菜单）
        self.lang_button = QtWidgets.QToolButton(self.centralwidget)
        self.lang_button.setMinimumSize(150, 20)
        self.lang_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.lang_button.setObjectName("lang_button")
        self.selfLayout.addWidget(self.lang_button, 9, 3, 1, 1)
        self.lang_menu = QtWidgets.QMenu(self.lang_button)
        self.lang_button.setMenu(self.lang_menu)

        # 添加主题选项
        for language in ('zh', 'en'):
            action = self.lang_menu.addAction(language)
            action.triggered.connect(  # type: ignore
                lambda checked, lang=language: SingletonConfig.apply_language(lang)
            )

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
        self.set_filepath_bt.setText(_translate("Settings", "SET..."))
        self.anim_checkBox.setText(_translate("Settings", "appear / pop / slide"))
        self.colorset_text.setText(_translate("Settings", "Tile Color:"))
        self.color_bt.setText(_translate("Settings", "Choose Color"))
        self.data_points_text.setText(_translate("Settings", "Data Points:"))
        self.spawnrate_text.setText(_translate("Settings", "4 Spawn Rate:"))
        self.theme_button.setText(_translate("Settings", "Set Theme"))
        self.demo_speed_text.setText(_translate("Settings", "Demo Speed:"))
        self.tile_font_size_text.setText(_translate("Settings", "Tile Font Size:"))
        self.anim_text.setText(_translate("Settings", "Do Animation:"))
        self.lang_button.setText("Language")  # 暂不翻译
        self.save_bt.setText(_translate("Settings", "SAVE"))
        self.pattern_combo.retranslateUi(self.tr("Select Formation"))
        self.target_combo.retranslateUi(self.tr("Target Tile"))
        self.pos_combo.retranslateUi(self.tr("Target Position"))

    def show_ColorDialog(self):
        config = SingletonConfig().config
        num = self.color_combo.currentText()
        current_color = config['colors'][int(np.log2(int(num))) - 1] if num != '' else '#000000'
        color_dialog = QtWidgets.QColorDialog(self)
        color_dialog.setWindowIcon(self.windowIcon())
        color = color_dialog.getColor(QtGui.QColor(current_color))
        if color.isValid() and num != '':
            num = int(np.log2(int(num))) - 1
            config['colors'][num] = color.name()

    @staticmethod
    def apply_theme(theme_name):
        if theme_name in theme_map:
            theme_colors = theme_map[theme_name]
            SingletonConfig().config['colors'] = theme_colors.copy() + ['#000000'] * 20
            SingletonConfig.tile_font_colors()

    def filepath_changed(self):
        options = QtWidgets.QFileDialog.Options()
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filepath = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", options=options)
        if filepath:
            self.filepath_edit.setText(filepath)

    def update_pos_combo_visibility(self, pattern):
        if pattern in ['444', 'LL']:
            if '2' in self.pos_combo.items:
                self.pos_combo.remove_item('2')
            self.pos_combo.show()
        elif pattern == 'L3':
            if '2' not in self.pos_combo.items:
                self.pos_combo.add_item('2')
            self.pos_combo.show()
        else:
            self.pos_combo.hide()

    def build_book(self):
        position = self.pos_combo.currentText
        pattern = self.pattern_combo.currentText
        target = self.target_combo.currentText
        pathname = self.filepath_edit.toPlainText()

        if pattern and target and pathname and os.path.exists(pathname):
            position = position if position else '0'
            config = SingletonConfig().config
            if pattern in ['444', 'LL', 'L3']:
                ptn = pattern + '_' + target + '_' + position
                config['filepath_map'][ptn] = [pathname, ]
                pathname = os.path.join(pathname, ptn + '_')
            else:
                ptn = pattern + '_' + target
                config['filepath_map'][ptn] = [pathname, ]
                pathname = os.path.join(pathname, ptn + '_')
            target = int(np.log2(int(target)))
            position = int(position)
            self.build_bt.setText(self.tr('Building...'))
            self.build_bt.setEnabled(False)
            self.Building_thread = BuildThread(pattern, target, position, pathname)
            self.Building_thread.finished.connect(self.on_build_finished)
            self.Building_thread.start()  # 启动计算线程

    def update_progress(self, current_step, total_steps):
        self.build_bt.setText(self.tr('Building...') + fr'({current_step}/{total_steps})')

    def on_build_finished(self):
        self.build_bt.setText(self.tr('BUILD'))
        self.build_bt.setEnabled(True)

    def demo_speed_changed(self):
        SingletonConfig().config['demo_speed'] = self.demo_speed_box.value()

    def font_size_changed(self):
        SingletonConfig().config['font_size_factor'] = self.tile_font_size_box.value()

    def update_data_points(self):
        SingletonConfig().config['data_points'] = int(self.data_points_box.text())

    def update_spawn_rate(self):
        SingletonConfig().config['4_spawn_rate'] = float(self.spawnrate_box.text().replace(',', '.'))

    def show_message(self):
        QtWidgets.QMessageBox.information(self, self.tr('Information'),
        self.tr('''This affects all modules of the program, make sure you know what you're doing'''))

    def save_all(self):
        SingletonConfig().config['do_animation'] = self.anim_checkBox.isChecked()
        SingletonConfig().config['data_points'] = int(self.data_points_box.text())
        SingletonConfig().config['4_spawn_rate'] = float(self.spawnrate_box.text().replace(',', '.'))
        SingletonConfig.save_config(SingletonConfig().config)


class BuildThread(QtCore.QThread):
    finished = QtCore.pyqtSignal()

    def __init__(self, pattern, target: int, position: int, pathname):
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
