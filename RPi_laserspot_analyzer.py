# -*- coding: utf-8 -*-

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from importlib import util

import datetime, time
import os, sys

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# Use picamera only if available
picamfound = util.find_spec("picamera") is not None
if picamfound:
    import picamera
    from picamera.array import PiBayerArray
else:
    print("No picamera module found, camera not available!")

version = 1.4


class MyEvent(QObject):
    my_event = pyqtSignal()


if picamfound:
    class MyCamera(picamera.PiCamera):
        def __init__(self, left=100, top=100, width=1024, height=768):
            super(MyCamera, self).__init__()

            self.prev_left, self.prev_top = left, top
            self.prev_width, self.prev_height = width, height

            self.imageres = [1024, 768]

            # set camera resolution, gain , sutter speed and framerate
            self.resolution = (self.imageres[0], self.imageres[1])
            # self.framerate = 33  # in Hz
            # self.shutter_speed = 500  # in us
            # self.exposure_mode = 'off'
            # self.iso = 300

        def my_start_preview(self):
            self.start_preview(
                fullscreen=False,
                window=(self.prev_left, self.prev_top, self.prev_width, self.prev_height)
            )


class PlotAnalyseCanvas(FigureCanvas):

    def __init__(self, pxl2um=None, width=5, height=4, dpi=100, parent=None):

        if pxl2um is None:
            self.pxl2um = MyMainWindow.pxl2um
        else:
            self.pxl2um = pxl2um

        # Initial parameter
        self.last_img, self.img_color = None, None
        self.init_x = dict(amp=None, mu=None, sig=None, off=None)  # amplitude, mu, sigma, offset
        self.init_y = dict(amp=None, mu=None, sig=None, off=None)  # amplitude, mu, sigma, offset
        self.order_x, self.order_y = 1, 1  # Order of gaussian distribution
        self.data_x, self.data_y = None, None  # Img data on chosen x and y line
        self.last_rslt = None  # Only sig +/- error
        self.data_x_rslt, self.data_y_rslt = None, None  # Complete fit results

        # Choose image color channel: 0 = red, 1 = green, b = blue
        self.color = 1

        # Miscellaneous
        self.now = None
        self.error_msg = None
        self.img_title = None

        # Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        # Don't know what the following is needed for

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    @staticmethod
    def gaussian(x, a, mu, sigma, offset):
        return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset

    @staticmethod
    def func(x, a, mu, sigma, offset, p=1):
        # return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)) ** p) - offset
        return a * np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)) ** p) + offset

    def calc_data(self):
        # Check self.last_img
        if type(self.last_img) is np.ndarray:
            img_data = self.last_img
        else:
            self.error_msg = "Could not calc from last_img = {}".format(self.last_img)
            print(self.error_msg)
            return self.error_msg

        print("Start Calculating...")
        if self.init_x['mu'] is None or self.init_y['mu'] is None:
            self.reset_mean_val(img_data)

        # Select image color channel (if not already given by data)
        if img_data.ndim == 2:
            self.img_color = img_data
        else:
            self.img_color = img_data[:, :, self.color]

        # Define x and y values from selected image
        x_x = np.linspace(0, len(self.img_color[0, :]), len(self.img_color[0, :]))
        x_y = self.img_color[self.init_y['mu'], :] / sum(self.img_color[self.init_y['mu'], :])
        self.data_x = np.array([x_x, x_y])

        y_x = np.linspace(0, len(self.img_color[:, 0]), len(self.img_color[:, 0]))
        y_y = self.img_color[:, self.init_x['mu']] / sum(self.img_color[:, self.init_x['mu']])
        self.data_y = np.array([y_x, y_y])

        # Initial fit parameter: amplitude, mu, sigma, offset, order
        # Todo find better sigma initial value
        # ToDo create Dialog to change these parameters
        if None in self.init_x.values():
            self.init_x = dict(amp=max(x_y), mu=self.init_x['mu'], sig=200, off=min(x_y))
        if None in self.init_y.values():
            self.init_y = dict(amp=max(x_y), mu=self.init_y['mu'], sig=200, off=min(y_y))

        self.print_init()

        # Filter empty sensor data before fitting
        self.temp_x = np.array([itm for itm in self.data_x.T if itm[1] != 0]).T
        self.temp_y = np.array([itm for itm in self.data_y.T if itm[1] != 0]).T

        try:

            # Residuals
            s_x, s_y = {}, {}
            # x line
            popt_x, pcov_x = curve_fit(lambda x, amp, mu, sig, off: self.func(x, amp, mu, sig, off, 1),
                                       *self.temp_x,
                                       p0=list(self.init_x.values()))
            s_x[1] = sum((self.func(self.temp_x[0], *popt_x) - self.temp_x[1]) ** 2)
            for q in range(2, 11):
                temp_popt_x, temp_pcov_x = curve_fit(lambda x, amp, mu, sig, off: self.func(x, amp, mu, sig, off, q),
                                                     *self.temp_x,
                                                     p0=list(self.init_x.values()))

                s_x[q] = sum((self.func(self.temp_x[0], *popt_x) - self.temp_x[1]) ** 2)

                if s_x[q] > s_x[q - 1]:
                    self.order_x = q - 1
                    break

                else:
                    popt_x, pcov_x = temp_popt_x, temp_pcov_x

            # y line
            popt_y, pcov_y = curve_fit(lambda y, amp, mu, sig, off: self.func(y, amp, mu, sig, off, 1),
                                       *self.temp_y,
                                       p0=list(self.init_y.values()))
            s_y[1] = sum((self.func(self.temp_y[0], *popt_y) - self.temp_y[1]) ** 2)
            for p in range(2, 11):
                temp_popt_y, temp_pcov_y = curve_fit(lambda y, amp, mu, sig, off: self.func(y, amp, mu, sig, off, p),
                                                     *self.temp_y,
                                                     p0=list(self.init_y.values()))

                s_y[p] = sum((self.func(self.temp_y[0], *popt_y) - self.temp_y[1]) ** 2)
                if s_y[p] > s_y[p - 1]:
                    self.order_y = p - 1
                    break
                else:
                    popt_y, pcov_y = temp_popt_y, temp_pcov_y

            print("Fit was successful!")

        except:
            idx = len(self.init_x)
            popt_x, popt_y = np.zeros(idx), np.zeros(idx)
            popt_x[2], popt_y[2] = 1, 1
            pcov_x, pcov_y = np.zeros((idx, idx)), np.zeros((idx, idx))
            print("Fit failed!")

        self.data_x_rslt = [popt_x, pcov_x]
        np.append(self.data_x_rslt[0], self.order_x)
        self.data_y_rslt = [popt_y, pcov_y]
        np.append(self.data_y_rslt[0], self.order_y)

        self.now = datetime.datetime.now()

        self.last_rslt = [2 * x * self.pxl2um for x in
                          [2 * self.data_x_rslt[0][2], 2 * np.sqrt(self.data_x_rslt[1][2, 2]),
                           2 * self.data_y_rslt[0][2], 2 * np.sqrt(self.data_y_rslt[1][2, 2])]]
        print("self.last_rslt = {}".format(self.last_rslt))
        print("Finished Calculating! Results:\n")
        self.print_init()
        print("final fit:\npopt_x = {}\npopt_y = {}".format(self.data_x_rslt, self.data_y_rslt))

        self.plot()

    def change_center(self, x, y):
        self.init_x['mu'] = x
        self.init_y['mu'] = y

    def change_color_channel(self, color=1):
        self.color = color

    def img_to_data(self):
        if self.last_img is None:
            self.error_msg = "No image selected yet"
            print(self.error_msg)
            return self.error_msg

        elif type(self.last_img) is str:
            self.img_title = self.last_img

            if self.last_img.endswith('.npz'):
                temp = np.load(self.last_img)
                self.last_img = temp.items()[0][1]

            else:
                self.last_img = mpimg.imread(self.last_img)


        elif type(self.last_img) is np.ndarray:
            pass

        else:
            self.error_msg = "Could not read image file: type(image) = {}".format(type(self.last_img))
            print(self.error_msg)
            return self.error_msg
        self.calc_data()

    def print_init(self):
        # For debugging purpose
        print("##########\n"
              "initial_fits:\n"
              "x = {},\ny = {}\n"
              "##########\n".format(self.init_x, self.init_y))

    def plot(self):
        print("Start Plotting")
        # Grid
        gs = gridspec.GridSpec(2, 3, hspace=.25, wspace=.3,
                               width_ratios=[4, 16, 1], height_ratios=[4, 1])

        # Clear last figure
        self.figure.clf()

        # Define axes in figure
        ax_image = self.figure.add_subplot(gs[:-1, 1:-1])
        ax_image_cb = self.figure.add_subplot(gs[:-1, -1])
        ax_y = self.figure.add_subplot(gs[:-1, 0], sharey=ax_image)
        ax_x = self.figure.add_subplot(gs[-1, 1:-1], sharex=ax_image)

        # Image, horizontal and vertical lines
        ax_image.autoscale(False)
        im = ax_image.imshow(self.img_color, cmap='jet', aspect='auto')

        ax_image.axhline(self.init_y['mu'], color='w')
        ax_image.axvline(self.init_x['mu'], color='w')

        ax_image.set_xlim(0, max(self.data_x[0]))
        ax_image.set_ylim(max(self.data_y[0]), 0)
        ax_image.set_xlabel('x in pixel')
        ax_image.set_ylabel('y in pixel')
        ax_image.set_title(self.img_title, y=1.12)
        ax_image.grid(color='w', alpha=0.5, linestyle='dashed', linewidth=0.5)

        # Colorbar
        ax_image_cb.set_label('Intensity')
        self.figure.colorbar(im, cax=ax_image_cb)

        # Second x and y axis in micrometer
        scld_xticks = np.around(self.pxl2um * np.arange(min(self.data_x[0]), max(self.data_x[0]), 250), 2)
        scld_yticks = np.around(self.pxl2um * np.arange(min(self.data_y[0]), max(self.data_y[0]), 250), 2)

        ax_image_x2 = ax_image.twiny()
        ax_image_x2.set_xticks(scld_xticks)
        ax_image_x2.set_xlabel("Micrometer")

        ax_image_y2 = ax_image.twinx()
        ax_image_y2.set_yticks(scld_yticks)
        ax_image_y2.invert_yaxis()
        ax_image_y2.set_ylabel("Micrometer")

        # Data and fits
        ax_y.autoscale(axis='y', enable=False)
        ax_y.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
        ax_y.plot(self.data_y[1], self.data_y[0], 'o', markersize=1, label="raw data")
        ax_y.plot(self.temp_y[1], self.temp_y[0], 'o', markersize=1, label="fitted data")
        ax_y.plot(self.func(self.data_y[0], *self.data_y_rslt[0]), self.data_y[0], label="fit")

        ax_x.autoscale(axis='x', enable=False)
        ax_x.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
        ax_x.plot(self.data_x[0], self.data_x[1], 'o', markersize=1, label="raw data")
        ax_x.plot(self.temp_x[0], self.temp_x[1], 'o', markersize=1, label="fitted data")
        ax_x.plot(self.data_x[0], self.func(self.data_x[0], *self.data_x_rslt[0]), label="fit")

        self.draw()

    def reset_mean_val(self, img_data):
        self.init_x['mu'] = int(np.round(img_data.shape[1] / 2))
        self.init_y['mu'] = int(np.round(img_data.shape[0] / 2))

    def show_img(self, pic=None):
        if pic is None:
            self.error_msg = "No image selected yet"
            print(self.error_msg)
            return self.error_msg
        elif type(pic) is picamera.array.PiBayerArray:
            plt.imshow(pic.demosaic())
            plt.show()
        elif type(pic) is np.ndarray:
            plt.imshow(pic)
            plt.show()


#############################
# Main Window
#############################
class MyMainWindow(QMainWindow):
    # Pixel to um
    pxl2um = 3760 / 2592

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)

        self.initUi()

    def initUi(self):
        # Screen Attributes
        screen_size = QDesktopWidget().availableGeometry()
        screen_res = [screen_size.width(), screen_size.height()]

        # Geometry
        self.my_left = round(0.05 * screen_res[0])
        self.my_top = round(0.05 * screen_res[0])
        self.my_height = round(0.8 * screen_res[1])
        self.my_width = self.my_height * 4 / 3  # round(0.8 * screen_res[0])

        self.form_widget = FormWidget(self)
        self.setCentralWidget(self.form_widget)

        # Menu bar
        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)
        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(exitAct)

        self.setGeometry(self.my_left, self.my_top, self.my_width, self.my_height)
        global version
        self.setWindowTitle('Another low budget Beam Profiler Version {}'.format(version))
        self.show()


#############################
# Form Widget
#############################
class FormWidget(QWidget):

    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        # Initial Parameters
        self.raw_bayer_data = None

        # Test Event
        self.sig = MyEvent()

        # Initialize camera
        preview = [round(x) for x in [parent.my_left, parent.my_top,
                                      0.8 * parent.my_width, 0.8 * parent.my_height]]
        if picamfound:
            self.camera = MyCamera(*preview)

        # Initialize UI
        self.init_ui(parent)

    def init_ui(self, parent):
        self.sig.my_event.connect(self.close)

        # FigureCanvas
        self.m = PlotAnalyseCanvas(None, 2 * parent.my_width, 2 * parent.my_height)
        self.m_toolbar = NavigationToolbar(self.m, self)

        # Slider to change Plot
        self.sp_m_x = QSlider(Qt.Horizontal)
        self.sp_m_x.setTickPosition(QSlider.TicksBelow)
        self.sp_m_x.sliderReleased.connect(self.change_mean_val)
        self.sp_m_y = QSlider(Qt.Vertical)
        self.sp_m_y.setTickPosition(QSlider.TicksRight)
        self.sp_m_y.sliderReleased.connect(self.change_mean_val)
        self.sp_m_y.setInvertedAppearance(True)

        lyt_plt = QGridLayout()
        lyt_plt.addWidget(self.m, 1, 0)
        lyt_plt.addWidget(self.sp_m_x, 0, 0)
        lyt_plt.addWidget(self.sp_m_y, 1, 1)

        # Navigation bar and latest results
        self.ln_edt_x = QLineEdit()
        self.ln_edt_x.setText("No results yet")
        self.ln_edt_wx = QLineEdit()
        self.ln_edt_wx.setText("No results yet")
        self.ln_edt_y = QLineEdit()
        self.ln_edt_y.setText("No results yet")
        self.ln_edt_wy = QLineEdit()
        self.ln_edt_wy.setText("No results yet")

        # Tables
        self.tbl = QTableWidget()
        self.tbl.setObjectName("table_view")
        self.tbl.setRowCount(2)
        self.tbl.setColumnCount(3)
        self.tbl.setMinimumHeight(100)
        tbl_col = ["2\u00b7\u03c3_rms", "FWHM", "Gaussian order"]
        tbl_row = ["x", "y"]
        self.tbl_indices = [[i, j] for i in range(self.tbl.rowCount()) for j in range(self.tbl.columnCount())]
        for i in range(self.tbl.columnCount()):
            self.tbl.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
            self.tbl.setHorizontalHeaderItem(i, QTableWidgetItem(tbl_col[i]))
        for i in range(self.tbl.rowCount()):
            self.tbl.verticalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
            self.tbl.setVerticalHeaderItem(i, QTableWidgetItem(tbl_row[i]))
        for idx in self.tbl_indices:
            self.tbl.setItem(*idx, QTableWidgetItem("No Results yet"))

        lyt_nav = QHBoxLayout()
        lyt_nav.addWidget(self.m_toolbar)
        # lyt_nav.addLayout(lyt_rslt)
        lyt_nav.addWidget(self.tbl)

        # Info and result box
        lbl_info = QLabel("Info")
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        lbl_rslt = QLabel("Result log:")
        self.txt_rslt = QTextEdit()
        self.txt_rslt.setReadOnly(True)
        self.txt_rslt.setMinimumHeight(100)

        lyt_txt = QGridLayout()
        lyt_txt.setSpacing(10)
        lyt_txt.addWidget(lbl_info, 0, 0)
        lyt_txt.addWidget(self.txt_info, 1, 0)
        lyt_txt.addWidget(lbl_rslt, 0, 1)
        lyt_txt.addWidget(self.txt_rslt, 1, 1)

        # Buttons
        self.btn_live_view = QPushButton("Start Live View")
        self.btn_live_view.setCheckable(True)
        self.btn_live_view.setChecked(False)
        self.btn_live_view.clicked.connect(lambda: self.start_live_view(self.btn_live_view))
        self.shortcut_live = QShortcut(QKeySequence("Ctrl+L"), self)
        self.shortcut_live.activated.connect(lambda: self.start_live_view(self.btn_live_view))

        btn_plot_file = QPushButton("Plot from File")
        btn_plot_file.clicked.connect(self.choose_file)

        btn_plot_pic = QPushButton("Take picture and plot")
        btn_plot_pic.clicked.connect(self.take_picture)
        self.shortcut_pic = QShortcut(QKeySequence("Ctrl+T"), self)
        self.shortcut_pic.activated.connect(self.take_picture)

        self.cbb_plot_cctr = QComboBox()
        self.cbb_plot_cctr.addItems(["red", "green", "blue"])
        self.cbb_plot_cctr.setCurrentIndex(self.m.color)
        self.cbb_plot_cctr.currentIndexChanged.connect(self.change_color)

        btn_show_img = QPushButton("Display converted raw image data")
        btn_show_img.clicked.connect(self.show_img)

        lyt_cctr = QHBoxLayout()
        lyt_cctr.addWidget(QLabel("Layer:"))
        lyt_cctr.addWidget(self.cbb_plot_cctr)

        btn_reset_slider = QPushButton("Center Slider")
        btn_reset_slider.clicked.connect(self.reset_slider)

        btn_save = QPushButton("Save last result")
        btn_save.clicked.connect(self.save)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(qApp.quit)

        lyt_btn = QHBoxLayout()
        lyt_btn.addWidget(self.btn_live_view)
        lyt_btn.addWidget(btn_plot_file)
        lyt_btn.addWidget(btn_plot_pic)
        lyt_btn.addLayout(lyt_cctr)
        lyt_btn.addWidget(btn_reset_slider)
        lyt_btn.addWidget(btn_show_img)
        lyt_btn.addWidget(btn_save)
        lyt_btn.addStretch(1)
        lyt_btn.addWidget(btn_close)

        # global layout
        lyt_glob = QVBoxLayout()
        lyt_glob.addLayout(lyt_plt)
        lyt_glob.addLayout(lyt_nav)
        lyt_glob.addLayout(lyt_txt)
        lyt_glob.addLayout(lyt_btn)

        self.setLayout(lyt_glob)

    def change_color(self):
        if self.m.last_img is None:
            self.txt_info.append("No image selected yet")
            return
        self.txt_info.append("Color {} selected".format(self.cbb_plot_cctr.currentText()))
        self.m.change_color_channel(self.cbb_plot_cctr.currentIndex())
        self.exec_calc()

    def choose_file(self):
        self.txt_info.append("Please choose a picture...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        my_img, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                "Images (*.png *.jpg *.tiff *.bmp *.npz);;All Files (*)",
                                                options=options)
        # Debug
        # my_img = "last_img.npz"
        if my_img:
            self.txt_info.append('{} chosen'.format(my_img))
            self.m.last_img = my_img
            self.exec_calc()

    def exec_calc(self):
        # Plot and analyse
        self.m.img_to_data()

        # Result handling
        rslt_str = "{}\n" \
                   "2*sigma_(x,rms) = ({:.2f} +/- {:.2f}) um\n" \
                   "2*sigma_(y,rms) = ({:.2f} +/- {:.2f}) um\n" \
            .format(self.m.now.strftime("%Y-%m-%d %H:%M:%S"), *self.m.last_rslt)
        self.txt_rslt.append(rslt_str)
        time.sleep(.5)

        # Adopt slider to image axis
        self.set_slider()

        # self.ln_edt_latest_rslt()
        self.write_to_table()

    def ln_edt_latest_rslt(self):
        FWHMx = [np.sqrt(2) * np.log(2) ** (1 / self.m.order_x) * x for x in self.m.last_rslt[:2]]
        FWHMy = [np.sqrt(2) * np.log(2) ** (1 / self.m.order_y) * y for y in self.m.last_rslt[2:]]
        self.ln_edt_x.setText('({:.2f} +/- {:.2f}) um'.format(*self.m.last_rslt[:2]))
        self.ln_edt_wx.setText('({:.2f} +/- {:.2f}) um'.format(*FWHMx))
        self.ln_edt_y.setText('({:.2f} +/- {:.2f}) um'.format(*self.m.last_rslt[2:]))
        self.ln_edt_wy.setText('({:.2f} +/- {:.2f}) um'.format(*FWHMy))

    def reset_slider(self):
        if self.m.last_img is None:
            self.txt_info.append("No image selected yet")
            return
        self.m.reset_mean_val(self.m.last_img)
        self.exec_calc()

    def save(self):
        self.txt_info.append("Please choose a saving path...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        my_path, _ = QFileDialog.getSaveFileName(self, "Save to", "last_img.npz",
                                                 "Numpy (*.npz);;All Files (*)", options=options)

        if not my_path:
            self.txt_info.append("Saving aborted!")
            return
        if self.m.img_color is None:
            self.txt_info.append("No data to be saved")
            return

        img = self.m.img_color
        linex_x, linex_y = self.m.data_x
        liney_x, liney_y = self.m.data_y
        fitx_popt, fitx_pcov = self.m.data_x_rslt
        fity_popt, fity_pcov = self.m.data_y_rslt

        np.savez_compressed(my_path, img=img,
                            linex_x=linex_x, linex_y=linex_y,
                            liney_x=liney_x, liney_y=liney_y,
                            fitx_popt=fitx_popt, fitx_pcov=fitx_pcov,
                            fity_popt=fity_popt, fity_pcov=fity_pcov)

        self.txt_info.append("Saving successful")

    def set_slider(self):
        self.sp_m_x.setMinimum(min(self.m.data_x[0]))
        self.sp_m_x.setMaximum(max(self.m.data_x[0]))
        self.sp_m_x.setValue(self.m.init_x['mu'])

        self.sp_m_y.setMinimum(min(self.m.data_y[0]))
        self.sp_m_y.setMaximum(max(self.m.data_y[0]))
        self.sp_m_y.setValue(self.m.init_y['mu'])

    def start_live_view(self, btn):
        if not picamfound:
            self.txt_info.append("Live view not available: No picamera module loaded")
            return

        if btn.isChecked():
            self.txt_info.append("Live view started")
            self.txt_info.append("Camera resolution: {}x{}".format(*self.camera.resolution))
            self.btn_live_view.setText("Stop Live View")
            self.camera.my_start_preview()

        else:
            self.txt_info.append("Live view stopped")
            self.camera.stop_preview()
            self.btn_live_view.setText("Start Live View")

    def show_img(self):
        if self.raw_bayer_data is None:
            self.txt_info.append("No raw image data found")
            return
        self.txt_info.append("Showing picture (this may take some time)...")
        self.m.show_img(self.raw_bayer_data)
        self.txt_info.append("Showing picture done.")

    def take_picture(self):
        if not picamfound:
            self.txt_info.append("Live view not available: No picamera module loaded")
            return
        # now = datetime.datetime.now()
        self.camera.my_start_preview()
        time.sleep(1)

        # PiBayerArray
        # http://picamera.readthedocs.io/en/release-1.13/api_array.html#pibayerarray
        self.raw_bayer_data = picamera.array.PiBayerArray(self.camera)
        self.camera.capture(self.raw_bayer_data, 'jpeg', bayer=True)

        self.camera.stop_preview()
        self.btn_live_view.setChecked(False)
        self.btn_live_view.setText("Start Live View")
        self.txt_info.append("Picture taken!")
        self.m.img_title = "Last Taken Picture (raw)"
        self.m.last_img = self.raw_bayer_data.array
        self.exec_calc()

    def test_emit_close_signal(self):
        self.sig.my_event.emit()

    def change_mean_val(self):
        self.m.init_x["mu"] = self.sp_m_x.value()
        self.m.init_y["mu"] = self.sp_m_y.value()
        if self.m.last_img is not None:
            self.exec_calc()

    def write_to_table(self):
        if self.m.last_rslt is None:
            self.txt_info.append("No results yet")
            return None
        FWHMx = [np.sqrt(2) * np.log(2) ** (1 / self.m.order_x) * x for x in self.m.last_rslt[:2]]
        FWHMy = [np.sqrt(2) * np.log(2) ** (1 / self.m.order_y) * y for y in self.m.last_rslt[2:]]
        sigm_x = '({:.2f} +/- {:.2f}) um'.format(*self.m.last_rslt[:2])
        fwhm_x = '({:.2f} +/- {:.2f}) um'.format(*FWHMx)
        sigm_y = '({:.2f} +/- {:.2f}) um'.format(*self.m.last_rslt[2:])
        fwhm_y = '({:.2f} +/- {:.2f}) um'.format(*FWHMy)
        temp_zip = zip(self.tbl_indices, [sigm_x, fwhm_x, self.m.order_x, sigm_y, fwhm_y, self.m.order_y])

        for idx in temp_zip:
            self.tbl.setItem(*idx[0], QTableWidgetItem(str(idx[1])))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyMainWindow()
    sys.exit(app.exec_())  # Mainloop

# ToDo Show complete fit result matrix
# ToDo Button to recenter slider
