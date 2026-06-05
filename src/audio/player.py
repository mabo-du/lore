from PyQt6.QtCore import QObject, pyqtSignal, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput


class AudioPlayer(QObject):
    """
    AudioPlayer wraps QMediaPlayer to provide precise playback control
    and frequent position updates for UI syncing.
    """

    position_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(int)
    state_changed = pyqtSignal(QMediaPlayer.PlaybackState)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_output)

        # Fire position_changed every 50ms for tight waveform/transcript sync
        self._player.positionChanged.connect(self.position_changed.emit)
        self._player.durationChanged.connect(self.duration_changed.emit)
        self._player.playbackStateChanged.connect(self.state_changed.emit)

        # In PyQt6, notify interval can be set via QTimer, but QMediaPlayer handles it intrinsically or via positionChanged
        # Actually QMediaPlayer in Qt6 emits positionChanged rapidly during playback.

    def load(self, file_path: str):
        self._player.setSource(QUrl.fromLocalFile(file_path))

    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def seek(self, ms: int):
        self._player.setPosition(ms)

    def position_ms(self) -> int:
        return self._player.position()

    def duration_ms(self) -> int:
        return self._player.duration()

    def set_volume(self, volume: float):
        """Volume between 0.0 and 1.0"""
        self._audio_output.setVolume(volume)
