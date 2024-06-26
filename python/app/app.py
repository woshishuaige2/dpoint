# Parts of this file were scaffolded from https://github.com/vispy/vispy/blob/main/examples/scene/realtime_data/ex03b_data_sources_threaded_loop.py
import datetime
import json
from pathlib import Path
import time
from typing import NamedTuple
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt

import vispy
from vispy import scene
from vispy.io import read_mesh
from vispy.scene import SceneCanvas, visuals
import vispy.app
from vispy.app import use_app
from vispy.util import quaternion
from vispy.visuals import transforms

import numpy as np
import queue
import multiprocessing as mp
from app.color_button import ColorButton
from app.filter import DpointFilter, blend_new_data

from app.marker_tracker import CameraReading, run_tracker
from app.monitor_ble import StopCommand, StylusReading, monitor_ble

CANVAS_SIZE = (1080, 1080)  # (width, height)
TRAIL_POINTS = 12000
USE_3D_LINE = (
    False  # If true, uses a lower quality GL line renderer that supports 3D lines
)

# Recording is only used for testing and evaluation of the system.
# When enabled the data from the IMU and camera are saved to disk, so they can be replayed
# offline with offline_ope.py and offline_playback,py.
recording_enabled = mp.Value("b", False)
app_start_datetime = datetime.datetime.now()
recording_timestamp = app_start_datetime.strftime("%Y%m%d_%H%M%S")


def append_line_point(line: np.ndarray, new_point: np.array):
    """Append new points to a line."""
    # There are many faster ways to do this, but this solution works well enough
    line[:-1, :] = line[1:, :]
    line[-1, :] = new_point


def get_line_color(line: np.ndarray):
    base_col = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    pos_z = line[:, [2]]
    return np.hstack(
        [
            np.tile(base_col, (line.shape[0], 1)),
            1 - np.clip(pos_z * 400, 0, 1),
        ]
    )


def get_line_color_from_pressure(pressure: float, color=(0, 0, 0, 1)):
    col = np.array(color, dtype=np.float32)
    col[3] *= np.clip(pressure, 0, 1)
    return col


class CameraUpdateData(NamedTuple):
    position_replace: list[np.ndarray]


class StylusUpdateData(NamedTuple):
    position: np.ndarray
    orientation: np.ndarray
    pressure: float


ViewUpdateData = CameraUpdateData | StylusUpdateData

class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE, vsync=False)
        self.canvas.measure_fps()
        self.canvas.connect(self.on_key_press)
        self.grid = self.canvas.central_widget.add_grid()

        self.view_top = self.grid.add_view(0, 0, bgcolor="white")
        self.view_top.camera = scene.TurntableCamera(
            up="z",
            fov=0,
            center=(0.10, 0.13, 0),
            elevation=90,
            azimuth=0,
            scale_factor=0.3,
        )
        vertices, faces, normals, texcoords = read_mesh("./python/mesh/pen.obj") 
        self.pen_mesh = visuals.Mesh(
            vertices, faces, color=(0.8, 0.8, 0.8, 1), parent=self.view_top.scene
        )
        self.pen_mesh.transform = transforms.MatrixTransform()
        self.line_color = (0, 0, 0, 1)

        pen_tip = visuals.XYZAxis(parent=self.pen_mesh)
        pen_tip.transform = transforms.MatrixTransform(
            vispy.util.transforms.scale([0.01, 0.01, 0.01])
        )

        self.line_data_pos = np.zeros((TRAIL_POINTS, 3), dtype=np.float32)
        self.line_data_col = np.zeros((TRAIL_POINTS, 4), dtype=np.float32)
        # agg looks much better than gl, but only works with 2D data.
        if USE_3D_LINE:
            self.trail_line = visuals.Line(
                width=1,
                parent=self.view_top.scene,
                method="gl",
            )
        else:
            self.trail_line = visuals.Line(
                width=3, parent=self.view_top.scene, method="agg", antialias=False
            )

        axis = scene.visuals.XYZAxis(parent=self.view_top.scene)
        axis.transform = transforms.MatrixTransform()
        axis.transform.scale([0.02, 0.02, 0.02])
        # This is broken for now, see https://github.com/vispy/vispy/issues/2363
        # grid = scene.visuals.GridLines(parent=self.view_top.scene)

    def update_data(self, new_data: ViewUpdateData):
        match new_data:
            case StylusUpdateData(
                position=pos, orientation=orientation, pressure=pressure
            ):
                orientation_quat = quaternion.Quaternion(*orientation).inverse()
                self.pen_mesh.transform.matrix = (
                    orientation_quat.get_matrix() @ vispy.util.transforms.translate(pos)
                )
                col = get_line_color_from_pressure(pressure, self.line_color)
                append_line_point(self.line_data_pos, pos)
                append_line_point(self.line_data_col, col)
            case CameraUpdateData(position_replace):
                if len(position_replace) == 0:
                    return
                view = self.line_data_pos[-len(position_replace) :, :]
                view[:, :] = blend_new_data(view, position_replace, 0.5)
                self.refresh_line()

    def refresh_line(self):
        # Skip rendering points where both ends have zero alpha
        pressure_mask = self.line_data_col[:, 3] > 0
        pressure_mask = (
            pressure_mask | np.roll(pressure_mask, 1) | np.roll(pressure_mask, -1)
        )
        pressure_mask[0:2] = True  # To ensure we always have at least one line segment
        pos = self.line_data_pos[pressure_mask, :]
        col = self.line_data_col[pressure_mask, :]
        self.trail_line.set_data(
            pos if USE_3D_LINE else pos[:, 0:2],
            color=col,
        )

    def clear_line(self):
        self.line_data_col[:, 3] *= 0
        self.refresh_line()

    def set_line_color(self, col: QtGui.QColor):
        self.line_color = col.getRgbF()  # (col.redF, col.greenF, col.blueF)

    def set_line_width(self, width: float):
        self.trail_line.set_data(width=width)

    def clear_last_stroke(self):
        diff = np.diff(
            (self.line_data_col[:, 3] > 0).astype(np.int8)
        )  # 1 when stroke starts, -1 when it ends
        start_indices = np.where(diff == 1)[0]
        if len(start_indices) > 0:
            last_stroke_index = start_indices[-1]
            print(TRAIL_POINTS - last_stroke_index)
            self.line_data_col[last_stroke_index:, 3] *= 0
        self.refresh_line()

    def on_key_press(self, e: vispy.app.canvas.KeyEvent):
        # if e.key == "R":
        #     if "Control" in e.modifiers:
        #         recording_enabled.value = True
        #         print("Recording enabled")
        #     else:
        #         recording_enabled.value = False
        #         print("Recording disabled")
        if e.key == "C":
            self.clear_line()
        elif e.key == "Z" and "Control" in e.modifiers:
            self.clear_last_stroke()


class MainWindow(QtWidgets.QMainWindow):
    closing = QtCore.pyqtSignal()

    def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()

        self._canvas_wrapper = canvas_wrapper
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        color_button = ColorButton("Line color")
        color_button.colorChanged.connect(canvas_wrapper.set_line_color)
        color_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear_button = QtWidgets.QPushButton("Clear (C)")
        clear_button.clicked.connect(canvas_wrapper.clear_line)
        clear_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        undo_button = QtWidgets.QPushButton("Undo (Ctrl+Z)")
        undo_button.clicked.connect(canvas_wrapper.clear_last_stroke)
        undo_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.line_width_label = QtWidgets.QLabel("")
        self.line_width_label.setMinimumWidth(80)
        self.line_width_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.line_width_slider.setRange(1, 20)
        self.line_width_slider.valueChanged.connect(self.line_width_changed)
        self.line_width_slider.setValue(2)
        bottom_toolbar = QtWidgets.QHBoxLayout()
        bottom_toolbar.addWidget(clear_button)
        bottom_toolbar.addWidget(undo_button)
        bottom_toolbar.addWidget(color_button)
        bottom_toolbar.addWidget(QtWidgets.QLabel("Thickness:"))
        bottom_toolbar.addWidget(self.line_width_slider)
        bottom_toolbar.addWidget(self.line_width_label)
        main_layout.addLayout(bottom_toolbar)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def line_width_changed(self, width):
        self.line_width_label.setText(str(width))
        self._canvas_wrapper.set_line_width(width)

    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        return super().closeEvent(event)


class QueueConsumer(QtCore.QObject):
    new_data = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal()

    def __init__(
        self,
        tracker_queue: "mp.Queue[CameraReading]",
        imu_queue: "mp.Queue[StylusReading]",
        parent=None,
    ):
        super().__init__(parent)
        self._should_end = False
        self._tracker_queue = tracker_queue
        self._imu_queue = imu_queue
        self._filter = DpointFilter(dt=1 / 145, smoothing_length=8, camera_delay=5)
        self._recorded_data_stylus = []
        self._recorded_data_camera = []

    def run_queue_consumer(self):
        print("Queue consumer is starting")
        samples_since_camera = 1000
        pressure_baseline = 0.017  # Approximate measured value for initial estimate
        pressure_avg_factor = 0.1  # Factor for exponential moving average
        pressure_range = 0.02
        pressure_offset = (
            0.002  # Offset so that small positive numbers are treated as zero
        )
        while True:
            if self._should_end:
                print("Data source saw that it was told to stop")
                break

            try:
                while self._tracker_queue.qsize() > 2:
                    self._tracker_queue.get()
                reading = self._tracker_queue.get_nowait()
                if recording_enabled.value:
                    self._recorded_data_camera.append(
                        (time.time_ns() // 1_000_000, reading)
                    )
                samples_since_camera = 0
                smoothed_tip_pos = self._filter.update_camera(
                    reading.position.flatten(), reading.orientation_mat
                )
                self.new_data.emit(CameraUpdateData(position_replace=smoothed_tip_pos))
            except queue.Empty:
                pass

            while self._imu_queue.qsize() > 0:
                reading = self._imu_queue.get()
                samples_since_camera += 1
                if samples_since_camera > 10:
                    continue
                if recording_enabled.value:
                    self._recorded_data_stylus.append(
                        (time.time_ns() // 1_000_000, reading)
                    )
                self._filter.update_imu(reading.accel, reading.gyro)
                position, orientation = self._filter.get_tip_pose()
                zpos = position[2]
                if zpos > 0.007:
                    # calibrate pressure baseline using current pressure reading
                    pressure_baseline = (
                        pressure_baseline * (1 - pressure_avg_factor)
                        + reading.pressure * pressure_avg_factor
                    )
                self.new_data.emit(
                    StylusUpdateData(
                        position=position,
                        orientation=orientation,
                        pressure=(
                            reading.pressure - pressure_baseline - pressure_offset
                        )
                        / pressure_range,
                    )
                )

        print("Queue consumer finishing")

        if self._recorded_data_stylus:
            file1 = Path(f"recordings/{recording_timestamp}/stylus_data.json")
            file1.parent.mkdir(parents=True, exist_ok=True)
            with file1.open("x") as f:
                json.dump(
                    [
                        dict(t=t, data=reading.to_json())
                        for t, reading in self._recorded_data_stylus
                    ],
                    f,
                )
            file2 = Path(f"recordings/{recording_timestamp}/camera_data_original.json")
            with file2.open("x") as f:
                json.dump(
                    [
                        dict(t=t, data=reading.to_json())
                        for t, reading in self._recorded_data_camera
                    ],
                    f,
                )

        self.finished.emit()

    def stop_data(self):
        print("Data source is quitting...")
        self._should_end = True


def run_tracker_with_queue(queue: mp.Queue, *args):
    run_tracker(lambda reading: queue.put(reading, block=False), *args)


def main():
    np.set_printoptions(
        precision=3, suppress=True, formatter={"float": "{: >5.2f}".format}
    )
    app = use_app("pyqt6")
    app.create()

    tracker_queue = mp.Queue()
    ble_queue = mp.Queue()
    ble_command_queue = mp.Queue()
    canvas_wrapper = CanvasWrapper()
    win = MainWindow(canvas_wrapper)
    win.resize(*CANVAS_SIZE)
    data_thread = QtCore.QThread(parent=win)

    queue_consumer = QueueConsumer(tracker_queue, ble_queue)
    queue_consumer.moveToThread(data_thread)

    camera_process = mp.Process(
        target=run_tracker_with_queue,
        args=(tracker_queue, recording_enabled, recording_timestamp),
        daemon=False,
    )
    camera_process.start()

    ble_process = mp.Process(
        target=monitor_ble, args=(ble_queue, ble_command_queue), daemon=False
    )
    ble_process.start()

    # update the visualization when there is new data
    queue_consumer.new_data.connect(canvas_wrapper.update_data)
    # start data generation when the thread is started
    data_thread.started.connect(queue_consumer.run_queue_consumer)
    # if the data source finishes before the window is closed, kill the thread
    queue_consumer.finished.connect(
        data_thread.quit, QtCore.Qt.ConnectionType.DirectConnection
    )
    # if the window is closed, tell the data source to stop
    win.closing.connect(
        queue_consumer.stop_data, QtCore.Qt.ConnectionType.DirectConnection
    )
    win.closing.connect(
        lambda: ble_command_queue.put(StopCommand()),
        QtCore.Qt.ConnectionType.DirectConnection,
    )
    # when the thread has ended, delete the data source from memory
    data_thread.finished.connect(queue_consumer.deleteLater)

    try:
        win.show()
        data_thread.start()
        app.run()
    finally:
        camera_process.terminate()
        ble_process.terminate()
    print("Waiting for data source to close gracefully...")
    data_thread.wait(1000)
