"""Simple GUI frontend for Jarvis with a waveform + glow visualizer.

This module is optional and only required if users want to run the desktop GUI. The GUI listens
for TTS playback notifications from the running `Jarvis` instance and displays a cinematic
waveform with glow while the assistant speaks.

Dependencies (optional):
- PySide6
- pyqtgraph

Run: `python -m jarvis.cli gui` or `uv run jarvis gui` (after installing optional deps)
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional
import concurrent.futures

import numpy as np

from loguru import logger

# Note: GUI dependencies are imported lazily inside `run_gui_with_config` so the package
# can be imported on systems where PySide6/pyqtgraph aren't available.

from .core.engine import Jarvis, JarvisConfig
from .core.audio_data import AudioMessage


# All GUI widget classes are defined lazily inside `run_gui_with_config` to avoid
# requiring GUI packages to be installed when importing this module.

def run_gui_with_config(config_path: str | None = None) -> int:  # pragma: no cover - GUI code
    """Entry point to run the GUI and hook into a Jarvis instance.

    Returns exit code like CLI commands (0 success, non-zero failure).
    """
    # Load config
    try:
        if config_path is None:
            config_path = "configs/jarvis_config.yaml"
        jarvis_config = JarvisConfig.from_yaml(str(config_path))
    except Exception as e:
        logger.exception(f"Failed to load config for GUI: {e}")
        return 1

    # Jarvis instance will be created after the QApplication exists to avoid
    # any accidental QWidget creation during model initialization.

    # Lazy-import GUI dependencies so the module can be imported without them installed
    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
        from PySide6.QtCore import QTimer, Qt, QPointF
        from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPainterPath
    except Exception as e:
        import sys

        print(
            "Missing GUI dependencies. Install with: 'python -m pip install PySide6 pyqtgraph' or 'pip install .[gui]'"
        )
        print(f"Import error: {e}")
        print(f"Python executable: {sys.executable}")
        return 1

    # Create QApplication early so any Qt widgets created during Jarvis construction
    # or initialization will have an active QApplication and won't raise errors.
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    # Install a Qt message handler to capture Qt messages and emit a Python stack trace;
    # this helps locate the source of 'QWidget: Must construct a QApplication before a QWidget'.
    try:
        from PySide6.QtCore import qInstallMessageHandler

        def _qt_msg_handler(msg_type, context, message):  # pragma: no cover - debug helper
            try:
                logger.error(f"Qt message: {message}")
                import traceback

                traceback.print_stack()
            except Exception:
                pass

        qInstallMessageHandler(_qt_msg_handler)
    except Exception:
        # If we can't install the handler, don't fail; just continue
        pass

    # Define GUI widgets now that dependencies are available
    class StringVisualizer(QWidget):
        """A lightweight cinematic horizontal string visualizer.

        Uses a per-segment RMS envelope (precomputed off the GUI thread) to drive
        a damped traveling sine wave along the horizontal "string". The widget
        draws multiple stroked layers for a glow effect and a progress indicator.
        """

        def __init__(self, parent: Optional[QWidget] = None) -> None:
            super().__init__(parent)
            self._envelope: Optional[np.ndarray] = None
            self._duration: float = 0.0
            self._start_time: Optional[float] = None
            self._sample_rate: int = 16000
            self._downsample_target = 300  # fewer points for a lightweight visual

            # Animation parameters
            self._phase = 0.0
            self._speed = 20.0  # phase speed (increased for snappier animation)
            self._spatial_freq = 0.02  # how quickly the sine changes across the string
            self._decay_tau = 1.8  # seconds for amplitude decay
            self._amplitude_px = 48  # maximum vertical displacement in pixels (slightly reduced)

            # Setup UI
            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.addStretch()  # push the label toward the bottom
            self.text_label = QLabel("")
            self.text_label.setAlignment(Qt.AlignCenter)
            # Larger font for better readability
            font = QFont()
            font.setPointSize(20)
            font.setBold(True)
            self.text_label.setFont(font)
            self.text_label.setStyleSheet("color: #7be1ff;")
            # Allow HTML rich text for token highlighting
            self.text_label.setTextFormat(Qt.RichText)
            layout.addWidget(self.text_label)

            # Timer for updates
            self.ui_timer = QTimer(self)
            self.ui_timer.setInterval(33)  # ~30 FPS
            self.ui_timer.timeout.connect(self._update)
            self.ui_timer.start()

        def set_precomputed(self, x: np.ndarray, envelope: np.ndarray, sample_rate: int, duration: float, text: str = "", word_timestamps: list[dict] | None = None) -> None:
            """Accept a precomputed RMS envelope and start the animation."""
            self._sample_rate = sample_rate
            env = np.asarray(envelope, dtype=np.float32)
            if env.size == 0:
                return
            # Normalize and smooth lightly
            env = env / max(1e-6, float(np.max(env)))
            if env.size > 3:
                env = np.convolve(env, np.ones(3) / 3.0, mode="same")
                env = env / max(1e-6, float(np.max(env)))

            self._envelope = env
            self._duration = duration
            # Only set start_time if not already set (avoids re-syncing when alignment arrives later)
            if self._start_time is None:
                self._start_time = time.time()
            self._phase = 0.0
            # If word-level timestamps are provided, use them for highlighting
            if word_timestamps:
                # Normalize timestamps to fractions of duration for robustness
                normalized: list[tuple[float, float]] = []
                tokens: list[str] = []
                for w in word_timestamps:
                    try:
                        start = float(w.get("start", 0.0))
                        end = float(w.get("end", 0.0))
                        # Clamp to duration
                        start = max(0.0, min(start, duration))
                        end = max(0.0, min(end, duration))
                        if end < start:
                            end = start
                        normalized.append((start / max(1e-6, duration), end / max(1e-6, duration)))
                        tokens.append(w.get("word", ""))
                    except Exception:
                        continue
                self._word_timestamps = list(word_timestamps)
                self._word_positions = normalized
                self._tokens = tokens
                self._last_highlight_index = -1
                logger.debug(f"Word positions (fractions): {self._word_positions}")
                self._update_tokens_display()
            else:
                # Prepare tokens and token timing positions (proportional to character lengths)
                self._word_timestamps = None
                try:
                    import html

                    tokens = [t for t in (str(text).split() if text else [])]
                    self._tokens = tokens
                    # Create start/end fractions for each token using character counts (including spaces)
                    total_chars = sum(len(t) for t in tokens) + max(0, len(tokens) - 1)
                    if total_chars == 0:
                        self._token_positions = []
                    else:
                        positions = []
                        cum = 0
                        for i, t in enumerate(tokens):
                            start = cum / total_chars
                            end = (cum + len(t)) / total_chars
                            positions.append((start, end))
                            cum += len(t) + 1  # include a space after each token except maybe last
                        self._token_positions = positions
                    self._last_highlight_index = -1
                    # Initially render the text unhighlighted
                    self._update_tokens_display()
                except Exception:
                    self._tokens = []
                    self._token_positions = []
                    self._last_highlight_index = -1
            self.update()

        def paintEvent(self, event) -> None:  # pragma: no cover - visual code
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), QColor(0, 0, 0))

            w = self.width()
            h = self.height()
            center_y = h // 2

            if self._envelope is None or self._start_time is None:
                # Idle breathing effect: small pulsating line
                t = time.time()
                base_amp = 6 * (0.5 + 0.5 * np.sin(t * 1.2))
                path = QPainterPath()
                path.moveTo(0, center_y)
                path.lineTo(w, center_y)
                pen = QPen(QColor(0, 120, 255, 40))
                pen.setWidth(10)
                painter.setPen(pen)
                painter.drawPath(path)
                painter.end()
                return

            # Compute elapsed and decay
            elapsed = time.time() - self._start_time
            frac = max(0.0, min(1.0, elapsed / self._duration)) if self._duration > 0 else 1.0
            decay = float(np.exp(-elapsed / self._decay_tau))

            env = self._envelope
            N = len(env)
            if N < 2:
                painter.end()
                return

            xs = np.linspace(0, w, N)
            self._phase += self._speed * (self.ui_timer.interval() / 1000.0)
            angles = self._phase + np.linspace(0, N * self._spatial_freq, N)

            disps = env * np.sin(angles) * decay * self._amplitude_px

            # Build path
            path = QPainterPath()
            path.moveTo(xs[0], center_y - disps[0])
            for i in range(1, N):
                path.lineTo(xs[i], center_y - disps[i])

            # Glow layers
            for width, color in ((18, QColor(0, 120, 255, 40)), (8, QColor(0, 160, 255, 120)), (2, QColor(255, 255, 255, 200))):
                pen = QPen(color)
                pen.setWidth(width)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawPath(path)

            # Progress indicator
            idx = int(frac * (N - 1))
            px = xs[idx]
            py = center_y - disps[idx]
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 200, 80, 140))
            painter.drawEllipse(QPointF(px, py), 12, 12)
            painter.setBrush(QColor(255, 200, 80, 220))
            painter.drawEllipse(QPointF(px, py), 5, 5)

            painter.end()

        def _update(self) -> None:
            # End the animation once duration elapses
            if self._start_time is None:
                return
            if self._duration > 0 and (time.time() - self._start_time) > (self._duration + 0.1):
                self._envelope = None
                self._start_time = None
                self.text_label.setText("")
            self.update()
            # Update token highlighting based on exact word timestamps if available
            if getattr(self, "_word_timestamps", None) and self._start_time is not None:
                elapsed = time.time() - self._start_time
                frac = elapsed / max(1e-6, self._duration)
                idx = None
                if getattr(self, "_word_positions", None):
                    for i, (s, e) in enumerate(self._word_positions):
                        if frac >= s and frac <= e:
                            idx = i
                            break
                    # If we've passed all words, highlight the last one
                    if idx is None and self._word_positions and frac > self._word_positions[-1][1]:
                        idx = len(self._word_positions) - 1
                else:
                    # Fallback to seconds-based check if normalized positions are missing
                    for i, w in enumerate(self._word_timestamps):
                        if elapsed >= float(w.get("start", 0.0)) and elapsed <= float(w.get("end", 0.0)):
                            idx = i
                            break
                    if idx is None and self._word_timestamps and elapsed > float(self._word_timestamps[-1].get("end", 0.0)):
                        idx = len(self._word_timestamps) - 1

                if idx != getattr(self, "_last_highlight_index", None):
                    self._last_highlight_index = idx
                    self._update_tokens_display()
            elif getattr(self, "_token_positions", None) and self._duration > 0 and self._start_time is not None:
                frac = max(0.0, min(1.0, (time.time() - self._start_time) / self._duration))
                idx = None
                for i, (s, e) in enumerate(self._token_positions):
                    if frac >= s and frac <= e:
                        idx = i
                        break
                if idx is None and self._token_positions:
                    idx = len(self._token_positions) - 1
                if idx != getattr(self, "_last_highlight_index", None):
                    self._last_highlight_index = idx
                    self._update_tokens_display()

        def _update_tokens_display(self) -> None:
            """Update the label to highlight the currently spoken token."""
            try:
                import html

                if not getattr(self, "_tokens", None):
                    self.text_label.setText("")
                    return
                parts: list[str] = []
                for i, tok in enumerate(self._tokens):
                    esc = html.escape(tok)
                    if i == getattr(self, "_last_highlight_index", -1):
                        parts.append(f"<span style='color:#7be1ff; font-weight:700'>{esc}</span>")
                    else:
                        parts.append(f"<span>{esc}</span>")
                html_text = " ".join(parts)
                self.text_label.setText(html_text)
            except Exception:
                # Fallback to plain text
                self.text_label.setText(" ".join(getattr(self, "_tokens", [])))

    class JarvisWindow(QMainWindow):
        def __init__(self, jarvis: Jarvis, audio_notification_queue: queue.Queue[AudioMessage]) -> None:
            super().__init__()
            self.jarvis = jarvis
            self.audio_notification_queue = audio_notification_queue

            self.setWindowTitle("Jarvis — Visualizer")
            self.setMinimumSize(800, 400)

            self.visualizer = StringVisualizer(self)
            self.setCentralWidget(self.visualizer)

            # Executor for expensive downsampling work (runs off the GUI thread)
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._pending_futures: list[concurrent.futures.Future] = []

            # Poll for audio notifications from the engine
            self.poll_timer = QTimer(self)
            self.poll_timer.setInterval(33)
            self.poll_timer.timeout.connect(self._poll_audio_queue)
            self.poll_timer.start()

        def _prepare_downsample(self, audio: np.ndarray, sample_rate: int, text: str, downsample_target: int):
            """Prepare a downsampled envelope for the visualizer and optionally compute ASR word timestamps.

            Returns a tuple (x, envelope, duration, text, sample_rate, word_timestamps) or None on failure.
            """
            try:
                if audio is None or getattr(audio, "size", 0) == 0:
                    return None
                # Ensure mono
                if audio.ndim > 1:
                    audio_proc = audio.mean(axis=1)
                else:
                    audio_proc = audio

                n = len(audio_proc)
                if n == 0:
                    return None

                factor = int(np.ceil(n / downsample_target)) if n > 0 else 1
                new_len = max(1, n // factor)
                trimmed = audio_proc[: new_len * factor]
                blocks = trimmed.reshape(new_len, factor)
                rms = np.sqrt(np.mean(blocks.astype(np.float32) ** 2, axis=1))
                max_rms = max(1e-6, float(np.max(rms)))
                envelope = (rms / max_rms).astype(np.float32)

                # Light smoothing
                if envelope.size > 3:
                    envelope = np.convolve(envelope, np.ones(3) / 3.0, mode="same")
                    envelope = envelope / max(1e-6, float(np.max(envelope)))

                x = np.linspace(0, len(envelope) / sample_rate, len(envelope))
                duration = float(len(audio_proc) / sample_rate)

                word_timestamps = None
                # Try ASR-based alignment if available
                try:
                    asr_model = getattr(self.jarvis, "_asr_model", None)
                    if asr_model is not None and hasattr(asr_model, "transcribe_with_word_timestamps"):
                        # Determine ASR sample rate
                        asr_sr = getattr(getattr(asr_model, "melspectrogram", None), "sample_rate", None)
                        if asr_sr is None:
                            asr_sr = getattr(asr_model, "sample_rate", None)
                        if asr_sr is not None:
                            asr_sr = int(asr_sr)
                            if sample_rate != asr_sr:
                                # resample audio_proc to asr_sr using linear interpolation
                                src = np.asarray(audio_proc, dtype=np.float32)
                                src_len = len(src)
                                target_len = int(round(src_len * (asr_sr / float(sample_rate))))
                                if target_len > 1 and src_len > 1:
                                    x_old = np.linspace(0, src_len - 1, src_len)
                                    x_new = np.linspace(0, src_len - 1, target_len)
                                    audio_resampled = np.interp(x_new, x_old, src).astype(np.float32)
                                else:
                                    audio_resampled = src
                            else:
                                audio_resampled = np.asarray(audio_proc, dtype=np.float32)

                            try:
                                word_timestamps = asr_model.transcribe_with_word_timestamps(
                                    audio_resampled, sample_rate=asr_sr
                                )
                                logger.debug(f"ASR alignment returned {len(word_timestamps)} words for text: {text}")
                            except Exception:
                                logger.exception("ASR alignment failed")
                except Exception:
                    logger.exception("Failed to attempt ASR alignment")

                return x, envelope, duration, text, sample_rate, word_timestamps
            except Exception as e:
                logger.exception(f"_prepare_downsample error: {e}")
                return None

        def _poll_audio_queue(self) -> None:
            # First, consume any new audio notifications and submit downsampling jobs
            try:
                while True:
                    audio_msg = self.audio_notification_queue.get_nowait()
                    if audio_msg.is_eos:
                        continue
                    # Immediate coarse envelope (fast) to kick-start the visualization
                    try:
                        a = audio_msg.audio
                        if a is not None and getattr(a, "size", 0) > 0:
                            if a.ndim > 1:
                                a_proc = a.mean(axis=1)
                            else:
                                a_proc = a
                            n = len(a_proc)
                            # Lightweight downsampling target to ~150 points
                            quick_target = 150
                            qfactor = max(1, n // quick_target)
                            if n >= qfactor:
                                trimmed_q = a_proc[: (n // qfactor) * qfactor]
                                blocks_q = trimmed_q.reshape(-1, qfactor)
                                rms_q = np.sqrt(np.mean(blocks_q.astype(np.float32) ** 2, axis=1))
                                max_rms_q = max(1e-6, float(rms_q.max()))
                                env_q = (rms_q / max_rms_q).astype(np.float32)
                                x_q = np.linspace(0, len(env_q) / self.jarvis._tts.sample_rate, len(env_q))
                                # Quick start (no timestamps yet)
                                try:
                                    self.visualizer.set_precomputed(x_q, env_q, self.jarvis._tts.sample_rate, float(len(a_proc) / self.jarvis._tts.sample_rate), audio_msg.text)
                                except Exception:
                                    pass
                    except Exception:
                        logger.exception("Quick envelope generation failed")
                    # Submit downsampling to the executor to avoid blocking the GUI thread
                    future = self._executor.submit(
                        self._prepare_downsample,
                        audio_msg.audio,
                        self.jarvis._tts.sample_rate,
                        audio_msg.text,
                        self.visualizer._downsample_target,
                    )
                    self._pending_futures.append(future)
            except queue.Empty:
                pass

            # Check for any completed downsampling work and apply immediately
            remaining: list[concurrent.futures.Future] = []
            for fut in self._pending_futures:
                if fut.done():
                    try:
                        res = fut.result()
                    except Exception as e:
                        logger.exception(f"Downsample worker future raised: {e}")
                        continue

                    if res is None:
                        continue

                    # Unpack optional word timestamps from worker res
                    if isinstance(res, tuple) and len(res) == 6:
                        x, y, duration, text, sr, word_ts = res
                    else:
                        try:
                            x, y, duration, text, sr = res
                            word_ts = None
                        except Exception:
                            logger.exception("Unexpected worker result format")
                            continue

                    try:
                        # Apply precomputed data on the GUI thread (we are in the Qt event loop)
                        # `y` is an RMS envelope normalized 0..1
                        self.visualizer.set_precomputed(x, y, sr, duration, text, word_timestamps=word_ts)
                    except Exception:
                        logger.exception("Failed to apply precomputed visualization data")
                else:
                    remaining.append(fut)
            self._pending_futures = remaining

    # Instantiate Jarvis now that QApplication exists
    try:
        jarvis = Jarvis.from_config(jarvis_config)
    except Exception as e:
        logger.exception(f"Failed to instantiate Jarvis for GUI: {e}")
        return 1

    # Queue to receive playback notifications
    audio_notification_queue: queue.Queue[AudioMessage] = queue.Queue()

    # Register a listener that enqueues the audio messages
    def enqueue_audio_msg(audio_msg: AudioMessage) -> None:
        try:
            audio_notification_queue.put_nowait(audio_msg)
        except Exception:
            logger.exception("Failed to enqueue audio message for GUI")

    jarvis.speech_player.register_playback_listener(enqueue_audio_msg)

    # Start the audio input system
    try:
        jarvis.audio_io.start_listening()
    except Exception:
        logger.exception("Failed to start audio input for GUI")

    # Optionally play the announcement if present
    play_announcement_text = jarvis.announcement

    # Keep the engine running in a background thread (it already started component threads in __init__)
    def engine_keepalive() -> None:
        while not jarvis.shutdown_event.is_set():
            time.sleep(0.05)

    threading.Thread(target=engine_keepalive, daemon=True).start()

    # Create and show the main window
    window = JarvisWindow(jarvis, audio_notification_queue)
    window.show()

    # Schedule announcement after the event loop starts so Qt is available
    if play_announcement_text:
        try:
            QTimer.singleShot(100, jarvis.play_announcement)
        except Exception:
            # If scheduling fails for any reason, call directly as a fallback
            try:
                jarvis.play_announcement()
            except Exception:
                logger.exception("Failed to play announcement")

    # Run the event loop
    try:
        rc = app.exec()
        return int(rc)
    except Exception:
        logger.exception("Failed to run QApplication")
        return 1
