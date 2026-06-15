import wave
import numpy as np
from tsdownsample import MinMaxDownsampler
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QResizeEvent, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal


class WaveformWidget(QWidget):
    """
    A custom QWidget that renders a large audio waveform using Level of Detail (LoD)
    downsampling via tsdownsample.
    """

    seek_requested = pyqtSignal(int)  # Emits position in ms

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setBackgroundRole(self.backgroundRole())
        self.setAutoFillBackground(True)

        self.audio_data = None
        self.downsampled_data = None
        self.sample_rate = 16000
        self.duration_ms = 0
        self.position_ms = 0

        # Colors (Dark Theme)
        self.bg_color = QColor("#1e1e1e")
        self.wave_color = QColor("#007acc")
        self.playhead_color = QColor("#ffaa00")

        palette = self.palette()
        palette.setColor(self.backgroundRole(), self.bg_color)
        self.setPalette(palette)

    def load_audio(self, wav_path: str):
        """Loads a 16kHz mono WAV file (output from normalise.py)"""
        try:
            with wave.open(wav_path, "rb") as wav:
                self.sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                self.duration_ms = int((n_frames / self.sample_rate) * 1000)

                # Read all frames and convert to numpy array
                frames = wav.readframes(n_frames)
                # Normalised audio is 16-bit PCM
                self.audio_data = np.frombuffer(frames, dtype=np.int16)
        except Exception as e:
            print(f"WaveformWidget: Failed to load audio: {e}")
            self.audio_data = None
            self.update()
            return

        self._update_downsampled_data()
        self.update()

    def set_position(self, ms: int):
        self.position_ms = ms
        self.update()

    def _update_downsampled_data(self):
        if self.audio_data is None or len(self.audio_data) == 0:
            return

        width = self.width()
        if width <= 0:
            return

        # We need min and max for each horizontal pixel
        # tsdownsample outputs interleaved min/max or similar depending on the downsampler.
        # MinMaxDownsampler downsamples to size `width * 2` (one min, one max per pixel bucket)
        downsampler = MinMaxDownsampler()
        # Create a dummy x array since tsdownsample requires it (can be arange)
        # Actually tsdownsample MinMaxDownsampler takes y and n_out.
        # It returns indices. We use indices to get the actual min/max values.

        # We want approximately 2 points (min, max) per pixel
        n_out = min(width * 2, len(self.audio_data))

        if n_out > 0:
            # downsample requires x, y, n_out — not just y
            x = np.arange(len(self.audio_data))
            indices = downsampler.downsample(x, self.audio_data, n_out=n_out)
            self.downsampled_data = self.audio_data[indices]
        else:
            self.downsampled_data = None

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self._update_downsampled_data()

    def mousePressEvent(self, event: QMouseEvent):
        if self.duration_ms > 0 and event.button() == Qt.MouseButton.LeftButton:
            x = event.position().x()
            pct = x / self.width()
            ms = int(pct * self.duration_ms)
            self.seek_requested.emit(ms)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), self.bg_color)

        width = self.width()
        height = self.height()

        if self.downsampled_data is not None and len(self.downsampled_data) > 0:
            pen = QPen(self.wave_color)
            pen.setWidth(1)
            painter.setPen(pen)

            # Draw waveform
            # The downsampled data has alternating min/max (or similar structure depending on n_out)
            # For simplicity, we draw lines connecting the points.
            # Scale y to fit widget height (int16 ranges from -32768 to 32767)

            x_step = width / len(self.downsampled_data)

            # Find max amplitude for normalization
            max_amp = np.max(np.abs(self.downsampled_data))
            if max_amp == 0:
                max_amp = 1

            center_y = height / 2

            for i in range(len(self.downsampled_data) - 1):
                x1 = int(i * x_step)
                x2 = int((i + 1) * x_step)

                # Scale -max_amp..max_amp to height..0
                y1 = center_y - (self.downsampled_data[i] / max_amp) * center_y
                y2 = center_y - (self.downsampled_data[i + 1] / max_amp) * center_y

                painter.drawLine(x1, int(y1), x2, int(y2))

        # Draw playhead
        if self.duration_ms > 0:
            playhead_x = int((self.position_ms / self.duration_ms) * width)
            pen = QPen(self.playhead_color)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(playhead_x, 0, playhead_x, height)
