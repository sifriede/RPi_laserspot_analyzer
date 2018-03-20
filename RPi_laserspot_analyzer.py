# -*- coding: utf-8 -*-

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
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

version = 1.3


class MyEvent(QObject):
    my_event = pyqtSignal()

if picamfound:
    class MyCamera(picamera.PiCamera):
        def __init__(self, left=100, top=100, width=1024, height=768):
            super(MyCamera, self).__init__()

            self.prev_left, self.prev_top = left, top
            self.prev_width, self.prev_height = width, height

            self.imageres = [1024, 768]
            # Pixel to um
            self.pxl2um = 3.76 / 2592

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
        self.last_img = None
        self.last_rslt = None
        self.img_color = None
        self.mean_img_x, self.mean_img_y = None, None
        self.sigma_x, self.sigma_y = 200, 200
        self.offset_x, self.offset_y = None, None
        self.data_x, self.data_y = None, None
        self.data_x_rslt, self.data_y_rslt = None, None

        # Choose image color channel: 0 = red, 1 = green, b = blue
        self.color = 1

        # Miscellaneous
        self.rslt_str = "No results yet"
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
    def gaussian(x, a, x0, sigma, offset):
        return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) - offset

    def change_center(self, x, y):
        self.mean_img_x = x
        self.mean_img_y = y
        # Debug
        print('New center: x,y = ({},{})'.format(x, y))

    def change_color_channel(self, color=1):
        self.color = color

    def reset_mean_val(self, img_data):
        self.mean_img_x = int(np.round(img_data.shape[1] / 2))
        self.mean_img_y = int(np.round(img_data.shape[0] / 2))

    def img_to_data(self):
        if self.last_img is None:
            self.error_msg = "No image selected yet"
            print(self.error_msg)
            return self.error_msg
        elif type(self.last_img) is str:
            self.img_title = self.last_img
            self.last_img = mpimg.imread(self.last_img)
            print(type(self.last_img))
        elif type(self.last_img) is np.ndarray:
            pass
        else:
            self.error_msg = "Could not read image file: type(image) = {}".format(type(self.last_img))
            print(self.error_msg)
            return self.error_msg
        self.calc_data()

    def calc_data(self):
        if type(self.last_img) is np.ndarray:
            img_data = self.last_img
        else:
            self.error_msg = "Could not calc from last_img = {}".format(self.last_img)
            print(self.error_msg)
            return self.error_msg

        print("Start Calculating...")
        if self.mean_img_x is None or self.mean_img_y is None:
            self.reset_mean_val(img_data)

        print("mean_img_x, mean_img_y = ({}, {})".format(self.mean_img_x, self.mean_img_y))

        self.img_color = img_data[:, :, self.color]

        x_x = np.linspace(0, len(self.img_color[0, :]), len(self.img_color[0, :]))
        x_y = self.img_color[self.mean_img_y, :] / sum(self.img_color[self.mean_img_y, :])
        self.data_x = [x_x, x_y]
        self.offset_x = min(x_y)

        y_x = np.linspace(0, len(self.img_color[:, 0]), len(self.img_color[:, 0]))
        y_y = self.img_color[:, self.mean_img_x] / sum(self.img_color[:, self.mean_img_x])
        self.data_y = [y_x, y_y]
        self.offset_y = min(y_y)

        try:
            popt_x, pcov_x = curve_fit(self.gaussian, self.data_x[0], self.data_x[1],
                                       p0=[1, self.mean_img_x, self.sigma_x, self.offset_x])
            popt_y, pcov_y = curve_fit(self.gaussian, self.data_y[0], self.data_y[1],
                                       p0=[1, self.mean_img_y, self.sigma_y, self.offset_y])
        except:
            popt_x, popt_y = [[0, 0, 1, 0], [0, 0, 1, 0]]
            pcov_x, pcov_y = np.zeros((4, 4)), np.zeros((4, 4))

        self.data_x_rslt = [popt_x, pcov_x]
        self.data_y_rslt = [popt_y, pcov_y]

        now = datetime.datetime.now()
        self.last_rslt = [2 * x * self.pxl2um for x in
                          [2 * self.data_x_rslt[0][2], 2 * np.sqrt(self.data_x_rslt[1][2, 2]),
                           2 * self.data_y_rslt[0][2], 2 * np.sqrt(self.data_y_rslt[1][2, 2])]]
        self.rslt_str = "{}\n" \
                        "sigma_(x,rms) = ({:.2f} +/- {:.2f}) um\n" \
                        "sigma_(y,rms) = ({:.2f} +/- {:.2f}) um\n" \
            .format(now.strftime("%Y-%m-%d %H:%M:%S"), *self.last_rslt)
        print("Finished Calculating! Results:\n")
        print("amp_x= {}, mu_x = {:.2f}, sigma_x = {:.2f}, offset_x = {:.2f}".format(*popt_x))
        print("pcov_x = {}".format(pcov_x))
        print("amp_y= {}, mu_y = {:.2f}, sigma_y = {:.2f}, offset_y = {:.2f}".format(*popt_y))
        print("pcov_y = {}".format(pcov_y))

        self.plot()

    def plot(self):
        # Grid
        gs = gridspec.GridSpec(2, 3, hspace=.25, wspace=.3,
                               width_ratios=[4, 16, 1], height_ratios=[4, 1])

        self.figure.clf()

        ax_image = self.figure.add_subplot(gs[:-1, 1:-1])
        ax_image.autoscale(False)
        im = ax_image.imshow(self.img_color, cmap='jet', aspect='auto')
        ax_image.axhline(self.mean_img_y, color='w')
        ax_image.axvline(self.mean_img_x, color='w')
        ax_image.set_title(self.img_title, y=1.12)
        ax_image.set_xlim(0, max(self.data_x[0]))
        ax_image.set_ylim(max(self.data_y[0]), 0)
        ax_image.set_xlabel('x in pixel')
        ax_image.set_ylabel('y in pixel')
        ax_image.grid(color='w', alpha=0.5, linestyle='dashed', linewidth=0.5)

        # Colorbar
        ax_image_cb = self.figure.add_subplot(gs[:-1, -1])
        ax_image_cb.set_label('Intensity')
        cbar = self.figure.colorbar(im, cax=ax_image_cb)
        # cbar.set_ticks([0, .25, .5, .75, 255])
        # cbar.set_ticklabels(['0', '25%', '50%', '75%', '100%'])

        # Second x and y axis in micrometer
        scld_xticks = np.around(self.pxl2um * np.arange(min(self.data_x[0]), max(self.data_x[0]), 100), 2)
        scld_yticks = np.around(self.pxl2um * np.arange(min(self.data_y[0]), max(self.data_y[0]), 100), 2)

        ax_image_x2 = ax_image.twiny()
        ax_image_x2.set_xticks(scld_xticks)
        ax_image_x2.set_xlabel("Micrometer")

        ax_image_y2 = ax_image.twinx()
        ax_image_y2.set_yticks(scld_yticks)
        ax_image_y2.invert_yaxis()
        ax_image_y2.set_ylabel("Micrometer")

        ax_y = self.figure.add_subplot(gs[:-1, 0], sharey=ax_image)
        ax_y.autoscale(axis='y', enable=False)
        ax_y.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
        ax_y.plot(self.data_y[1], self.data_y[0])
        ax_y.plot(self.gaussian(self.data_y[0], *self.data_y_rslt[0]), self.data_y[0])
        #
        ax_x = self.figure.add_subplot(gs[-1, 1:-1], sharex=ax_image)
        ax_x.autoscale(axis='x', enable=False)
        ax_x.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)
        ax_x.plot(self.data_x[0], self.data_x[1])
        ax_x.plot(self.data_x[0], self.gaussian(self.data_x[0], *self.data_x_rslt[0]))

        self.draw()


class MyMainWindow(QWidget):
    # Pixel to um
    pxl2um = 3760 / 2592

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)

        # Screen Attributes
        screen_size = QDesktopWidget().availableGeometry()
        screen_res = [screen_size.width(), screen_size.height()]

        # Geometry
        self.my_left = round(0.05 * screen_res[0])
        self.my_top = round(0.05 * screen_res[0])
        self.my_height = round(0.8 * screen_res[1])
        self.my_width = self.my_height * 4 / 3  # round(0.8 * screen_res[0])

        # Test Event
        self.sig = MyEvent()

        # Initialize camera
        preview = [round(x) for x in [self.my_left, self.my_top,
                                      0.8 * self.my_width, 0.8 * self.my_height]]
        if picamfound:
            self.camera = MyCamera(*preview)

        # Initialize UI
        self.init_ui()

    def init_ui(self):
        self.sig.my_event.connect(self.close)

        # FigureCanvas
        self.m = PlotAnalyseCanvas(None, self.my_width, 2 * self.my_height)
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

        lbl_font = QFont("Helvetica", 12)
        lbl_x = QLabel("2\u00b7\u03c3<sub>x,rms</sub>:")
        lbl_x.setFont(lbl_font)
        lbl_wx = QLabel("FWHM<sub>x</sub>:")
        lbl_wx.setFont(lbl_font)
        lbl_y = QLabel("2\u00b7\u03c3<sub>y,rms</sub>:")
        lbl_y.setFont(lbl_font)
        lbl_wy = QLabel("FWHM<sub>y</sub>:")
        lbl_wy.setFont(lbl_font)

        lyt_rslt = QGridLayout()
        lyt_rslt.addWidget(lbl_x, 0, 0)
        lyt_rslt.addWidget(self.ln_edt_x, 0, 1)
        lyt_rslt.addWidget(lbl_wx, 0, 2)
        lyt_rslt.addWidget(self.ln_edt_wx, 0, 3)
        lyt_rslt.addWidget(lbl_y, 1, 0)
        lyt_rslt.addWidget(self.ln_edt_y, 1, 1)
        lyt_rslt.addWidget(lbl_wy, 1, 2)
        lyt_rslt.addWidget(self.ln_edt_wy, 1, 3)

        lyt_nav = QHBoxLayout()
        lyt_nav.addWidget(self.m_toolbar)
        lyt_nav.addLayout(lyt_rslt)

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

        btn_plot_file = QPushButton("Plot from File")
        btn_plot_file.clicked.connect(self.choose_file)

        btn_plot_pic = QPushButton("Take picture and plot")
        btn_plot_pic.clicked.connect(self.take_picture)

        self.cbb_plot_cctr = QComboBox()
        self.cbb_plot_cctr.addItems(["red", "green", "blue"])
        self.cbb_plot_cctr.setCurrentIndex(self.m.color)
        self.cbb_plot_cctr.currentIndexChanged.connect(self.change_color)

        lyt_cctr = QHBoxLayout()
        lyt_cctr.addWidget(QLabel("Layer:"))
        lyt_cctr.addWidget(self.cbb_plot_cctr)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.test_emit_close_signal)

        btn_reset_slider = QPushButton("Center Slider")
        btn_reset_slider.clicked.connect(self.reset_slider)

        lyt_btn = QHBoxLayout()
        lyt_btn.addWidget(self.btn_live_view)
        lyt_btn.addWidget(btn_plot_file)
        lyt_btn.addWidget(btn_plot_pic)
        lyt_btn.addLayout(lyt_cctr)
        lyt_btn.addWidget(btn_reset_slider)
        lyt_btn.addStretch(1)
        lyt_btn.addWidget(btn_close)

        lyt_vrt = QVBoxLayout()
        lyt_vrt.addLayout(lyt_plt)
        lyt_vrt.addLayout(lyt_nav)
        lyt_vrt.addLayout(lyt_txt)
        lyt_vrt.addLayout(lyt_btn)

        self.setLayout(lyt_vrt)
        self.setGeometry(self.my_left, self.my_top, self.my_width, self.my_height)
        global version
        self.setWindowTitle('Another low budget Beam Profiler Version {}'.format(version))
        self.show()

    def test_emit_close_signal(self):
        self.sig.my_event.emit()

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

    def choose_file(self):
        self.txt_info.append("Please choose a picture...")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        my_img, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                "Images (*.png *.jpg *.tiff *.bmp);;All Files (*)", options=options)
        if my_img:
            self.txt_info.append('{} chosen'.format(my_img))
            self.m.last_img = my_img
            self.exec_calc()

    def set_slider(self):
        self.sp_m_x.setMinimum(min(self.m.data_x[0]))
        self.sp_m_x.setMaximum(max(self.m.data_x[0]))
        self.sp_m_x.setValue(self.m.mean_img_x)

        self.sp_m_y.setMinimum(min(self.m.data_y[0]))
        self.sp_m_y.setMaximum(max(self.m.data_y[0]))
        self.sp_m_y.setValue(self.m.mean_img_y)

    def reset_slider(self):
        if self.m.last_img is None:
            self.txt_info.append("No image selected yet")
            return
        self.m.reset_mean_val(self.m.last_img)
        self.exec_calc()

    def take_picture(self):
        if not picamfound:
            self.txt_info.append("Live view not available: No picamera module loaded")
            return
        now = datetime.datetime.now()
        self.camera.my_start_preview()
        time.sleep(1)

        # PiBayerArray
        # http://picamera.readthedocs.io/en/release-1.13/api_array.html#pibayerarray
        my_pic_dat = picamera.array.PiBayerArray(self.camera)
        self.camera.capture(my_pic_dat, 'rgb')
        self.camera.stop_preview()
        self.txt_info.append("Picture taken!")
        self.m.img_title = "Last Taken Picture (raw)"
        self.m.last_img = my_pic_dat.array
        self.exec_calc()

    def change_color(self):
        if self.m.last_img is None:
            self.txt_info.append("No image selected yet")
            return
        self.txt_info.append("Color {} selected".format(self.cbb_plot_cctr.currentText()))
        self.m.change_color_channel(self.cbb_plot_cctr.currentIndex())
        self.exec_calc()

    def change_mean_val(self):
        self.m.mean_img_x = self.sp_m_x.value()
        self.m.mean_img_y = self.sp_m_y.value()
        if self.m.last_img is not None:
            self.exec_calc()

    def set_latest_rslt(self):
        FWHM = [np.sqrt(2 * np.log(2)) * x for x in self.m.last_rslt]
        self.ln_edt_x.setText('({:.2f} +/- {:.2f}) um'.format(*self.m.last_rslt[:2]))
        self.ln_edt_wx.setText('({:.2f} +/- {:.2f}) um'.format(*FWHM[:2]))
        self.ln_edt_y.setText('({:.2f} +/- {:.2f}) um'.format(*self.m.last_rslt[2:]))
        self.ln_edt_wy.setText('({:.2f} +/- {:.2f}) um'.format(*FWHM[2:]))

    def exec_calc(self):
        self.m.img_to_data()
        self.txt_rslt.append(self.m.rslt_str)
        time.sleep(.5)
        self.set_slider()
        self.set_latest_rslt()
        self.txt_rslt.append(str(self.m.last_rslt))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MyMainWindow()
    sys.exit(app.exec_())  # Mainloop

# ToDo Show complete fit result matrix
# ToDo Button to recenter slider
