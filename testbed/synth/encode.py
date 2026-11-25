"""H.264 MP4 writing through ffmpeg (falls back to OpenCV's MPEG-4 Part 2
writer when ffmpeg is not available; such files play in fewer players)."""
from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                                   # pip install imageio-ffmpeg ships a static binary
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class VideoWriter:
    """Context manager: `with VideoWriter(path, w, h, fps) as vw: vw.write(bgr)`."""

    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int = 21,
                 preset: str = "veryfast", ffmpeg: Optional[str] = None):
        self.path = Path(path)
        self.size = (width, height)
        self.fps = fps
        exe = ffmpeg or find_ffmpeg()
        self._proc = self._cv = None
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
                   "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                   "-g", str(int(round(fps * 2))), str(self.path)]
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            warnings.warn("ffmpeg not found: writing MPEG-4 Part 2 with OpenCV "
                          "(pip install imageio-ffmpeg for H.264)")
            self._cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size)

    def write(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1::-1] != self.size:
            raise ValueError(f"frame is {frame_bgr.shape[1::-1]}, writer expects {self.size}")
        if self._proc:
            self._proc.stdin.write(np.ascontiguousarray(frame_bgr).tobytes())
        else:
            self._cv.write(frame_bgr)

    def close(self) -> None:
        if self._proc:
            self._proc.stdin.close()
            if self._proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
            self._proc = None
        elif self._cv is not None:
            self._cv.release()
            self._cv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
