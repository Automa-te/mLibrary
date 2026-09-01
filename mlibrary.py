"""
mlibrary.py  —  Portable Media Library  v2.0
Architecture: painter-based virtual canvas (no per-card QWidget).
Requires: PyQt5, opencv-python, Pillow, numpy
"""

import sys, os, sqlite3, hashlib, zipfile, shutil, subprocess, csv
import json, time, threading, traceback, logging, io, re, tempfile, atexit
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("mlibrary")

def _excepthook(t, v, tb):
    log.critical("CRASH:\n%s", "".join(traceback.format_exception(t, v, tb)))
    print(f"\033[91m[CRASH] {t.__name__}: {v}\033[0m", file=sys.stderr)
    sys.__excepthook__(t, v, tb)
sys.excepthook = _excepthook

def _thread_excepthook(a):
    log.critical("THREAD CRASH (thread=%s):\n%s",
        getattr(a.thread, 'name', '?'),
        "".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))
threading.excepthook = _thread_excepthook

# ── PyQt5 ────────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QAbstractScrollArea,
    QToolBar, QAction, QSlider, QFileDialog, QMessageBox,
    QDialog, QDialogButtonBox, QProgressBar, QCheckBox, QMenu,
    QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QFormLayout, QStyle, QLayout, QSizePolicy, QToolTip,
    QScrollArea, QFrame, QTextEdit, QInputDialog, QSplitter, QComboBox,
    QCompleter, QGridLayout
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSize, QPoint, QRect,
    QRectF, pyqtSlot, QObject, QStringListModel
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QFontMetrics,
    QPalette, QBrush, QCursor, QIcon
)
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io as _io
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

# ── Constants ─────────────────────────────────────────────────────────────────
APP_VERSION      = "2.0.0"
LIBRARY_FILE     = "mlibrary.lib"
CONFIG_FILE      = "library.config"
THUMB_BASE       = 250
THUMB_QUALITY    = 75
VIDEO_THUMB_COUNT = 10
HOVER_MS         = 250
PRELOAD_ROWS     = 3    # extra rows to fetch beyond visible screen
CELL_PAD         = 10
LABEL_H          = 34
BADGE_R          = 14
RATING_ROW_H     = 16       # height of the star/heart row overlay
RATING_ICON_SZ   = 12       # px per star/heart glyph when clickable
RATING_MIN_CELL  = 145      # below this cell size, ratings become view-only text
                             # (10 icons x RATING_ICON_SZ + margins must fit in one row
                             # without the star and heart groups overlapping)
GROUP_HEADER_H   = 34        # height of a collapsible group header band

IMAGE_EXTS = {'.jpg','.jpeg','.webp','.png','.bmp','.gif','.tiff','.tif'}
VIDEO_EXTS = {'.mp4','.mkv','.m4v','.webm','.mpeg','.mpg'}
ZIP_EXTS   = {'.zip'}
ALL_EXTS   = IMAGE_EXTS | VIDEO_EXTS | ZIP_EXTS

# ── Session temp directory ────────────────────────────────────────────────────
#
# Used for extracting zip contents to preview/export as PDF. Everything
# under this directory is deleted when the app closes normally (atexit) —
# nothing extracted for a preview is ever written into the library folder
# or left behind on disk after the session ends.

_SESSION_TMP_DIR: Optional[str] = None

def get_session_tmp_dir() -> str:
    global _SESSION_TMP_DIR
    if _SESSION_TMP_DIR is None:
        _SESSION_TMP_DIR = tempfile.mkdtemp(prefix="mlibrary_")
        log.info("Session temp dir: %s", _SESSION_TMP_DIR)
    return _SESSION_TMP_DIR

def _cleanup_session_tmp_dir():
    global _SESSION_TMP_DIR
    if _SESSION_TMP_DIR and os.path.isdir(_SESSION_TMP_DIR):
        try:
            shutil.rmtree(_SESSION_TMP_DIR, ignore_errors=True)
            log.info("Cleaned up session temp dir: %s", _SESSION_TMP_DIR)
        except Exception:
            log.exception("Failed to clean up session temp dir")
    _SESSION_TMP_DIR = None

atexit.register(_cleanup_session_tmp_dir)

# ── Palette ───────────────────────────────────────────────────────────────────
DARK_BG  = "#1a1b1e"; PANEL_BG = "#25262b"; CARD_BG  = "#2c2d33"
CARD_HOV = "#35363d"; ACCENT   = "#5865f2"; ACCENT2  = "#7289da"
TEXT_PRI = "#e0e0e0"; TEXT_SEC = "#8b8c9a"; BORDER   = "#3a3b42"
SEL_COL  = "#5865f2"; DANGER   = "#ed4245"; WARN     = "#fee75c"
SUCCESS  = "#57f287"

BADGE_COLS = {'image':'#4a90d9','video':'#e74c3c','zip':'#f39c12','folder':'#8e44ad'}
BADGE_ICONS= {'image':'🖼','video':'▶','zip':'📦','folder':'📁'}

# ── Global text scale ──────────────────────────────────────────────────────
#
# A single mutable scale factor (1.0-2.0) applied to every font-size in the
# app: the QSS stylesheet (build_app_style below), the painter-based
# gallery's raw QFont objects (GalleryCanvas.set_text_scale), and the many
# per-widget inline setStyleSheet() calls scattered through the dialogs
# (via the fs() helper). Persisted in AppConfig so it survives restarts.

UI_TEXT_SCALE = 1.0   # updated by MainWindow._set_text_scale; read by fs()

def fs(px: int) -> int:
    """Scale a base pixel font size by the current global UI_TEXT_SCALE.
    Used everywhere a font-size is written into an f-string stylesheet,
    so a single scale change reaches every dialog consistently."""
    return max(1, round(px * UI_TEXT_SCALE))

def ws(px: int) -> int:
    """
    Scale a base pixel WIDTH/HEIGHT by the current global UI_TEXT_SCALE.
    Used for buttons/inputs sized with setFixedWidth/setFixedSize that
    hold text — without this, a fixed pixel box stays the same size while
    its font grows, clipping the label down to an unreadable sliver (the
    bug this was added to fix). Callers generally want setMinimumWidth
    with this value rather than setFixedWidth, so Qt can still grow the
    widget further if the actual rendered text needs more room than the
    scaled estimate assumes.
    """
    return max(px, round(px * UI_TEXT_SCALE))

def build_app_style(scale: float = 1.0) -> str:
    """
    Builds the full QSS stylesheet at the given text scale. Called once at
    startup and again whenever the user changes the text-size setting;
    the result is applied via QApplication.setStyleSheet() so it cascades
    to every window without each dialog needing to re-apply it itself.

    Padding/min-height on interactive controls (buttons, inputs, combo
    boxes) also scale, not just font-size — otherwise larger text at 200%
    would visually clip against a control sized for 100% text.
    """
    f13, f12, f11 = fs(13), fs(12), fs(11)
    pad_v  = max(6, round(6 * scale))     # vertical padding on buttons/inputs
    pad_h  = max(10, round(12 * scale))   # horizontal padding
    ctrl_h = max(16, round(16 * scale))   # checkbox/radio indicator size
    return f"""
QMainWindow,QDialog{{background:{DARK_BG};color:{TEXT_PRI};}}
QWidget{{background:{DARK_BG};color:{TEXT_PRI};font-family:'Segoe UI',Arial,sans-serif;font-size:{f13}px;}}
QToolBar{{background:{PANEL_BG};border-bottom:1px solid {BORDER};padding:4px 8px;spacing:6px;}}
QToolBar QToolButton{{background:transparent;color:{TEXT_PRI};border:none;border-radius:4px;padding:{pad_v}px {pad_h}px;font-size:{f13}px;}}
QToolBar QToolButton:hover{{background:{CARD_HOV};}}
QPushButton{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {BORDER};border-radius:5px;padding:{pad_v}px {pad_h+2}px;font-size:{f13}px;}}
QPushButton:hover{{background:{CARD_HOV};border-color:{ACCENT2};}}
QPushButton:pressed{{background:{ACCENT};border-color:{ACCENT};}}
QPushButton:checked{{background:{ACCENT};border-color:{ACCENT};color:white;font-weight:bold;}}
QPushButton:checked:hover{{background:{ACCENT2};border-color:{ACCENT2};}}
QPushButton#accent{{background:{ACCENT};border-color:{ACCENT};font-weight:bold;}}
QPushButton#accent:hover{{background:{ACCENT2};}}
QPushButton#danger{{background:{DANGER};border-color:{DANGER};color:white;font-weight:bold;}}
QLineEdit{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {BORDER};border-radius:5px;padding:{pad_v-1}px {pad_h-2}px;font-size:{f13}px;selection-background-color:{ACCENT};}}
QLineEdit:focus{{border-color:{ACCENT};}}
QScrollBar:vertical{{background:{PANEL_BG};width:8px;border-radius:4px;}}
QScrollBar::handle:vertical{{background:{BORDER};border-radius:4px;min-height:20px;}}
QScrollBar::handle:vertical:hover{{background:{ACCENT2};}}
QScrollBar:horizontal{{background:{PANEL_BG};height:8px;border-radius:4px;}}
QScrollBar::handle:horizontal{{background:{BORDER};border-radius:4px;}}
QScrollBar::add-line,QScrollBar::sub-line{{width:0;height:0;}}
QStatusBar{{background:{PANEL_BG};color:{TEXT_SEC};border-top:1px solid {BORDER};font-size:{f12}px;padding:2px 8px;}}
QMenu{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {BORDER};border-radius:6px;padding:4px;font-size:{f13}px;}}
QMenu::item{{padding:{pad_v}px 20px {pad_v}px 12px;border-radius:3px;}}
QMenu::item:selected{{background:{ACCENT};}}
QMenu::separator{{height:1px;background:{BORDER};margin:3px 8px;}}
QLabel{{background:transparent;color:{TEXT_PRI};font-size:{f13}px;}}
QProgressBar{{background:{PANEL_BG};border:1px solid {BORDER};border-radius:5px;text-align:center;color:{TEXT_PRI};font-size:{f12}px;}}
QProgressBar::chunk{{background:{ACCENT};border-radius:4px;}}
QSlider::groove:horizontal{{background:{PANEL_BG};height:4px;border-radius:2px;}}
QSlider::handle:horizontal{{background:{ACCENT};width:14px;height:14px;margin:-5px 0;border-radius:7px;}}
QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:2px;}}
QTreeWidget{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {BORDER};border-radius:5px;alternate-background-color:{CARD_BG};font-size:{f13}px;}}
QTreeWidget::item:selected{{background:{ACCENT};}}
QListWidget{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {BORDER};border-radius:5px;font-size:{f13}px;}}
QListWidget::item:selected{{background:{ACCENT};}}
QTextEdit{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {BORDER};border-radius:5px;padding:6px;font-size:{f13}px;}}
QTextEdit:focus{{border-color:{ACCENT};}}
QSplitter::handle{{background:{BORDER};}}
QSplitter::handle:hover{{background:{ACCENT2};}}
QComboBox{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {BORDER};border-radius:5px;padding:{pad_v-1}px {pad_h-2}px;font-size:{f13}px;}}
QComboBox:focus{{border-color:{ACCENT};}}
QComboBox QAbstractItemView{{background:{PANEL_BG};color:{TEXT_PRI};selection-background-color:{ACCENT};border:1px solid {BORDER};font-size:{f13}px;}}
QGroupBox{{color:{TEXT_SEC};border:1px solid {BORDER};border-radius:6px;margin-top:8px;padding-top:8px;font-size:{f12}px;}}
QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 4px;}}
QCheckBox{{color:{TEXT_PRI};spacing:6px;font-size:{f13}px;}}
QCheckBox::indicator{{width:{ctrl_h}px;height:{ctrl_h}px;border:1px solid {BORDER};border-radius:3px;background:{PANEL_BG};}}
QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}
QRadioButton{{color:{TEXT_PRI};spacing:6px;font-size:{f13}px;}}
QRadioButton::indicator{{width:{ctrl_h}px;height:{ctrl_h}px;border:1px solid {BORDER};border-radius:{ctrl_h//2}px;background:{PANEL_BG};}}
QRadioButton::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}
QToolTip{{background:{PANEL_BG};color:{TEXT_PRI};border:1px solid {ACCENT2};padding:4px 8px;font-size:{f12}px;}}
"""

# Default-scale stylesheet, used as the module-level constant everywhere
# existing code refers to APP_STYLE — kept for any straggler reference,
# but the live, scale-aware stylesheet is applied app-wide via
# QApplication.setStyleSheet(build_app_style(...)) in main()/MainWindow.
APP_STYLE = build_app_style(1.0)

# ── Thumbnail utilities ───────────────────────────────────────────────────────

def md5_file(path: str, chunk=1 << 20) -> str:
    h = hashlib.md5()
    try:
        with open(path, 'rb') as f:
            for c in iter(lambda: f.read(chunk), b''):
                h.update(c)
        return h.hexdigest()
    except Exception:
        return ""

def _pil_to_jpeg(img: Image.Image) -> bytes:
    img.thumbnail((THUMB_BASE, THUMB_BASE), Image.LANCZOS)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    buf = _io.BytesIO()
    img.save(buf, format='JPEG', quality=THUMB_QUALITY, optimize=True)
    return buf.getvalue()

def thumb_image(path: str) -> Optional[bytes]:
    try:
        with Image.open(path) as img:
            return _pil_to_jpeg(img)
    except Exception:
        return None

def thumb_video(path: str) -> List[bytes]:
    out = []
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: cap.release(); return []
        for i in range(VIDEO_THUMB_COUNT):
            pos = int(total * i / VIDEO_THUMB_COUNT)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                b = _pil_to_jpeg(pil)
                if b: out.append(b)
        cap.release()
    except Exception:
        pass
    return out

def thumb_folder_mosaic(image_paths: List[str]) -> Optional[bytes]:
    """
    Builds a THUMB_BASE x THUMB_BASE 2x2 mosaic from up to 4 images —
    used as a folder's thumbnail when it directly contains images, so the
    folder card shows a preview of its contents rather than a generic
    folder icon. Each image is center-cropped to a square and placed in
    one quadrant. If fewer than 4 images are available, only that many
    quadrants are filled (top-left first, reading order); the rest stay
    transparent/background so it still reads clearly as a folder preview.
    Returns None if no images could be opened at all.
    """
    if not image_paths:
        return None
    paths = image_paths[:4]
    half = THUMB_BASE // 2
    # CARD_BG is a "#RRGGBB" hex string — convert to an (R,G,B) tuple, which
    # is what PIL's Image.new expects for an RGB-mode background fill.
    bg_rgb = tuple(int(CARD_BG.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    canvas = Image.new('RGB', (THUMB_BASE, THUMB_BASE), bg_rgb)

    positions = [(0,0), (half,0), (0,half), (half,half)]
    placed = 0
    for path, (px, py) in zip(paths, positions):
        try:
            with Image.open(path) as img:
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')
                # Center-crop to a square so the quadrant fills cleanly
                # without distortion, then resize down to the quadrant size.
                w, h = img.size
                side = min(w, h)
                left = (w - side) // 2
                top  = (h - side) // 2
                img = img.crop((left, top, left + side, top + side))
                img = img.resize((half, half), Image.LANCZOS)
                canvas.paste(img, (px, py))
                placed += 1
        except Exception:
            continue   # leave that quadrant as background — not a hard failure

    if placed == 0:
        return None

    buf = _io.BytesIO()
    canvas.save(buf, format='JPEG', quality=THUMB_QUALITY, optimize=True)
    return buf.getvalue()

def natural_sort_key(name: str):
    """
    Natural/numeric sort key: '002.jpg' sorts before '010.jpg', unlike
    plain alphabetical sort which would put '010.jpg' before '002.jpg'.
    Splits the string into alternating text/number chunks.
    """
    return [int(chunk) if chunk.isdigit() else chunk.lower()
            for chunk in re.split(r'(\d+)', name)]

def analyze_zip_contents(path: str) -> Tuple[List[bytes], int]:
    """
    Reads a zip archive once and returns (thumbnails, item_count):
      - thumbnails: a single-element list containing the JPEG cover
        thumbnail (the first top-level image inside the zip, natural-
        sorted), or [] if the zip has no readable top-level images. Only
        one is stored — the normal gallery grid's cover is the only
        consumer; the zip folder preview window extracts and shows every
        image on demand instead of relying on pre-stored thumbnails, so
        storing more than one here would just be wasted space in the
        library file for no reader.
      - item_count: total number of TOP-LEVEL entries in the zip — a
        subfolder inside the zip counts as ONE item, its own contents are
        not counted individually (matches Folder counting below, and the
        spec: "count the folder but not the child contents").
    Returns ([], 0) if the zip can't be read at all.
    """
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            names = zf.namelist()
            top_level_dirs = set()
            top_level_images = []
            for n in names:
                if n.endswith('/'):
                    continue   # explicit directory entries — handled via prefix below
                # Determine the top-level component of this path
                parts = n.split('/', 1)
                if len(parts) == 1:
                    # File directly at zip root
                    if Path(n).suffix.lower() in IMAGE_EXTS:
                        top_level_images.append(n)
                else:
                    # File inside a subfolder — the subfolder itself is
                    # the "item", its contents don't count separately
                    top_level_dirs.add(parts[0])

            item_count = len(top_level_dirs) + len([
                n for n in names if '/' not in n and not n.endswith('/')
            ])

            top_level_images.sort(key=natural_sort_key)
            thumbs = []
            if top_level_images:
                try:
                    img = Image.open(_io.BytesIO(zf.read(top_level_images[0])))
                    data = _pil_to_jpeg(img)
                    if data:
                        thumbs.append(data)
                except Exception:
                    pass
            return thumbs, item_count
    except Exception:
        return [], 0

def thumb_zip(path: str) -> Optional[bytes]:
    """Cover thumbnail only (first top-level image) — kept as a thin
    wrapper around analyze_zip_contents for callers that only need the
    single cover image, not the full multi-thumbnail set or item count."""
    thumbs, _count = analyze_zip_contents(path)
    return thumbs[0] if thumbs else None

def jpeg_to_qimage(data: bytes) -> Optional[QImage]:
    """Decode JPEG bytes → QImage (thread-safe). Returns None on failure."""
    if not data:
        return None
    img = QImage()
    img.loadFromData(data)
    return img if not img.isNull() else None

# ── Image enhancement (sharpen + contrast) ────────────────────────────────────
#
# Used by both PDF export (Mild/Strong/Custom/Off) and the zip reader
# (Mild/Strong/Off, no Custom — matches the A1/A2/Off reader mockup).
# Factors are RELATIVE to each image's own current values (this is how
# PIL's ImageEnhance and UnsharpMask already work — a contrast factor of
# 1.10 means "10% more contrast than this image already has", not a fixed
# absolute target), so a naturally sharp/contrasty image is barely touched
# while a blurry one gets the same proportional lift.

ENHANCE_PRESETS = {
    # level_name: (unsharp_percent, contrast_factor, noise_reduction_percent)
    # Used by the export dialog's Mild/Strong buttons. Kept noise-free —
    # the reader view's A2 gets its own +15% noise reduction on top of
    # this same sharpen/contrast baseline via READER_ENHANCE_PRESETS below,
    # without affecting the export dialog's Strong preset.
    'mild':   (108, 1.08, 0),
    'strong': (140, 1.35, 0),
}
READER_ENHANCE_PRESETS = {
    # level_name: (unsharp_percent, contrast_factor, noise_reduction_percent)
    # Used by the zip reader / image viewer's A1/A2 buttons specifically.
    # Same sharpen/contrast baseline as ENHANCE_PRESETS, but A2 also
    # applies mild (15%) noise reduction, per spec — scoped to the reader
    # only so the export dialog's "Strong" preset is unaffected.
    'mild':   (108, 1.08, 0),
    'strong': (140, 1.35, 15),
}
ENHANCE_CUSTOM_RANGE = {
    'sharpen':  (100, 250),   # UnsharpMask percent — 100 = no-op
    'contrast': (1.00, 2.00), # ImageEnhance.Contrast factor — 1.00 = no-op
    'noise':    (0, 100),     # noise reduction strength, percent — 0 = no-op
}

def reduce_noise(img: Image.Image, strength_percent: float) -> Image.Image:
    """
    Blend-based noise reduction: apply a mild smoothing filter, then blend
    the smoothed result back with the original at strength_percent/100 —
    0% leaves the image untouched, 100% is fully smoothed. This keeps the
    control as a simple percentage (matching sharpen/contrast's existing
    style) rather than exposing filter-specific parameters.
    MedianFilter is used for the smoothing pass since it suppresses
    speckle/grain noise while preserving edges better than a plain
    Gaussian blur would.
    """
    if not strength_percent or strength_percent <= 0:
        return img
    strength = max(0.0, min(1.0, strength_percent / 100.0))
    smoothed = img.filter(ImageFilter.MedianFilter(size=3))
    if strength >= 1.0:
        return smoothed
    return Image.blend(img, smoothed, strength)

def enhance_image(img: Image.Image, sharpen_percent: float,
                   contrast_factor: float, noise_percent: float = 0) -> Image.Image:
    """
    Apply noise reduction, then an unsharp mask, then a contrast
    adjustment — in that order, since sharpening noisy detail first would
    amplify the very noise this is meant to remove. All three are no-ops
    at their neutral values (noise_percent=0, sharpen_percent=100,
    contrast_factor=1.0) so callers can pass 'off' straight through
    without branching.
    """
    if noise_percent and noise_percent > 0:
        img = reduce_noise(img, noise_percent)
    if sharpen_percent and sharpen_percent > 100:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2, percent=int(sharpen_percent), threshold=3))
    if contrast_factor and contrast_factor != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img

def enhance_params_for_level(level: str, custom_sharpen: float = None,
                              custom_contrast: float = None,
                              custom_noise: float = None):
    """Resolve a level name ('mild'/'strong'/'custom'/'off') to
    (sharpen_percent, contrast_factor, noise_percent). Unknown levels
    behave as 'off'."""
    if level == 'off' or level is None:
        return 100, 1.0, 0
    if level == 'custom':
        return (custom_sharpen or 100), (custom_contrast or 1.0), (custom_noise or 0)
    return ENHANCE_PRESETS.get(level, (100, 1.0, 0))

def qimage_to_pixmap(img: QImage) -> QPixmap:
    return QPixmap.fromImage(img)

def placeholder_qimage(w: int, h: int, icon: str, bg: str = CARD_BG) -> QImage:
    img = QImage(w, h, QImage.Format_RGB32)
    img.fill(QColor(bg))
    p = QPainter(img)
    p.setPen(QColor(TEXT_SEC))
    f = QFont("Segoe UI", max(w // 8, 10))
    p.setFont(f)
    p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, icon)
    p.end()
    return img

# ── Config ────────────────────────────────────────────────────────────────────

class AppConfig:
    DEFAULTS = {"library_path":"","thumb_scale":"1.0","window_geometry":"",
                "text_scale":"1.0"}
    def __init__(self, path: str):
        self.path = path; self.data = dict(self.DEFAULTS); self.load()
    def load(self):
        self.data = dict(self.DEFAULTS)
        if os.path.exists(self.path):
            try:
                with open(self.path,'r',encoding='utf-8') as f:
                    self.data.update(json.load(f))
            except Exception: pass
    def save(self):
        try:
            with open(self.path,'w',encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception: pass
    def get(self, k, default=""):
        return self.data.get(k, default or self.DEFAULTS.get(k,""))
    def set(self, k, v):
        self.data[k] = str(v); self.save()

# ── Database ──────────────────────────────────────────────────────────────────

class LibraryDB:
    SCHEMA_TABLES = """
    CREATE TABLE IF NOT EXISTS files(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL, rel_path TEXT NOT NULL UNIQUE,
        abs_path TEXT, file_type TEXT NOT NULL,
        file_size INTEGER, checksum TEXT,
        width INTEGER, height INTEGER, duration REAL,
        date_added TEXT, date_modified TEXT,
        is_deleted INTEGER DEFAULT 0, parent_id INTEGER REFERENCES files(id),
        stars INTEGER DEFAULT 0, hearts INTEGER DEFAULT 0,
        item_count INTEGER
    );
    CREATE TABLE IF NOT EXISTS thumbnails(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        frame_index INTEGER DEFAULT 0, data BLOB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tags(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        category TEXT NOT NULL, subcategory TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tag_vocab(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL, subcategory TEXT NOT NULL,
        UNIQUE(category, subcategory)
    );
    CREATE TABLE IF NOT EXISTS sources(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        abs_path TEXT NOT NULL UNIQUE,
        recursive INTEGER DEFAULT 0,
        date_added TEXT
    );
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
    """

    SCHEMA_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_fp  ON files(rel_path);
    CREATE INDEX IF NOT EXISTS idx_ft  ON files(file_type);
    CREATE INDEX IF NOT EXISTS idx_fd  ON files(is_deleted);
    CREATE INDEX IF NOT EXISTS idx_fst ON files(stars);
    CREATE INDEX IF NOT EXISTS idx_fh  ON files(hearts);
    CREATE INDEX IF NOT EXISTS idx_tf  ON thumbnails(file_id,frame_index);
    CREATE INDEX IF NOT EXISTS idx_tg  ON tags(file_id);
    CREATE INDEX IF NOT EXISTS idx_tgc ON tags(category,subcategory);
    CREATE INDEX IF NOT EXISTS idx_tvc ON tag_vocab(category,subcategory);
    """
    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        self._local   = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local,'conn',None):
            c = sqlite3.connect(self.lib_path, check_same_thread=False, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA cache_size=-32000")
            c.execute("PRAGMA mmap_size=268435456")
            c.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = c
        return self._local.conn

    @property
    def conn(self): return self._conn()

    def _init_db(self):
        # Phase 1: create tables if they don't exist yet. Note that
        # CREATE TABLE IF NOT EXISTS is a no-op on an older library that
        # already has a 'files' table — it will NOT add new columns like
        # stars/hearts. That's what phase 2 (the ALTER TABLE migration)
        # is for, and it MUST run before phase 3 creates indexes on those
        # columns, or CREATE INDEX fails with "no such column".
        self.conn.executescript(self.SCHEMA_TABLES)
        self.conn.execute("INSERT OR IGNORE INTO meta VALUES('version','2.1')")
        self.conn.execute("INSERT OR IGNORE INTO meta VALUES('created',?)",
                          (datetime.now().isoformat(),))

        # Phase 2: migrate older libraries created before stars/hearts/
        # item_count existed
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(files)")]
        if 'stars' not in cols:
            self.conn.execute("ALTER TABLE files ADD COLUMN stars INTEGER DEFAULT 0")
        if 'hearts' not in cols:
            self.conn.execute("ALTER TABLE files ADD COLUMN hearts INTEGER DEFAULT 0")
        if 'item_count' not in cols:
            self.conn.execute("ALTER TABLE files ADD COLUMN item_count INTEGER")
        self.conn.commit()

        # Phase 2.5: backfill tag_vocab from any tags already assigned to
        # files in older libraries (tag_vocab is a NEW table as of this
        # version — before it existed, a tag only "existed" if some file
        # used it). This makes every previously-used tag visible in the
        # new Tag Manager without losing anything.
        self.conn.execute("""
            INSERT OR IGNORE INTO tag_vocab(category, subcategory)
            SELECT DISTINCT category, subcategory FROM tags
        """)
        self.conn.commit()

        # Phase 3: create indexes now that stars/hearts columns are guaranteed
        # to exist, whether from a fresh table or the migration just above.
        self.conn.executescript(self.SCHEMA_INDEXES)
        self.conn.commit()

    def file_count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE is_deleted=0").fetchone()[0]

    def get_library_stats(self) -> dict:
        """
        Aggregate stats for the Library Information dialog. Kept as one
        method (rather than several small ones) so the dialog can show
        everything from a single, consistent snapshot of the database.
        """
        c = self.conn
        stats = {}

        # Counts by type (active, non-deleted)
        rows = c.execute(
            "SELECT file_type, COUNT(*) as cnt FROM files "
            "WHERE is_deleted=0 GROUP BY file_type").fetchall()
        by_type = {r['file_type']: r['cnt'] for r in rows}
        stats['images'] = by_type.get('image', 0)
        stats['videos'] = by_type.get('video', 0)
        stats['zips']   = by_type.get('zip', 0)
        stats['folders']= by_type.get('folder', 0)
        stats['total_files'] = stats['images'] + stats['videos'] + stats['zips']

        # Size (folders have no file_size of their own — excluded naturally
        # since file_size is NULL for them)
        row = c.execute(
            "SELECT COALESCE(SUM(file_size),0) as total, "
            "COALESCE(AVG(file_size),0) as avg_size "
            "FROM files WHERE is_deleted=0 AND file_type != 'folder'").fetchone()
        stats['total_size'] = row['total']
        stats['avg_size']   = row['avg_size']

        # Tags / vocabulary
        stats['tag_count'] = c.execute(
            "SELECT COUNT(*) FROM tag_vocab").fetchone()[0]
        stats['category_count'] = c.execute(
            "SELECT COUNT(DISTINCT category) FROM tag_vocab").fetchone()[0]
        stats['untagged_files'] = c.execute(
            "SELECT COUNT(*) FROM files f WHERE f.is_deleted=0 "
            "AND f.file_type != 'folder' AND NOT EXISTS "
            "(SELECT 1 FROM tags t WHERE t.file_id=f.id)").fetchone()[0]

        # Ratings
        stats['starred_files'] = c.execute(
            "SELECT COUNT(*) FROM files WHERE is_deleted=0 AND stars>0").fetchone()[0]
        stats['hearted_files'] = c.execute(
            "SELECT COUNT(*) FROM files WHERE is_deleted=0 AND hearts>0").fetchone()[0]

        # Deleted (missing) — what Compact would clean up
        stats['deleted_files'] = c.execute(
            "SELECT COUNT(*) FROM files WHERE is_deleted=1").fetchone()[0]

        # Date range (date_modified — see grouping/sorting for why not date_added)
        row = c.execute(
            "SELECT MIN(date_modified) as oldest, MAX(date_modified) as newest "
            "FROM files WHERE is_deleted=0 AND file_type != 'folder'").fetchone()
        stats['oldest_date'] = row['oldest']
        stats['newest_date'] = row['newest']

        # Sources tracked (for rescan)
        stats['source_count'] = c.execute(
            "SELECT COUNT(*) FROM sources").fetchone()[0]

        # Top 10 largest files — id/abs_path included so the dialog can
        # make each entry clickable (open in default app).
        stats['largest_files'] = c.execute(
            "SELECT id, filename, file_size, abs_path FROM files "
            "WHERE is_deleted=0 AND file_type != 'folder' AND file_size IS NOT NULL "
            "ORDER BY file_size DESC LIMIT 10").fetchall()

        # Library (.lib) file size on disk
        try:
            stats['db_file_size'] = os.path.getsize(self.lib_path)
        except OSError:
            stats['db_file_size'] = 0

        return stats

    def get_files(self, parent_id=None, offset=0, limit=None) -> List:
        q = "SELECT * FROM files WHERE is_deleted=0"
        p = []
        if parent_id is None: q += " AND parent_id IS NULL"
        else: q += " AND parent_id=?"; p.append(parent_id)
        q += " ORDER BY file_type DESC, filename COLLATE NOCASE"
        if limit: q += f" LIMIT {limit} OFFSET {offset}"
        return self.conn.execute(q, p).fetchall()

    def search_files(self, name="", inc_tags=None, exc_tags=None,
                     parent_id=None, rating_filters=None) -> List:
        q = "SELECT DISTINCT f.* FROM files f WHERE f.is_deleted=0"
        p = []
        if parent_id is None: q += " AND f.parent_id IS NULL"
        else: q += " AND f.parent_id=?"; p.append(parent_id)
        if name.strip():
            q += " AND f.filename LIKE ?"; p.append(f"%{name.strip()}%")
        for cat, sub in (inc_tags or []):
            q += " AND EXISTS(SELECT 1 FROM tags t WHERE t.file_id=f.id AND t.category=? AND t.subcategory=?)"
            p += [cat, sub]
        for cat, sub in (exc_tags or []):
            q += " AND NOT EXISTS(SELECT 1 FROM tags t WHERE t.file_id=f.id AND t.category=? AND t.subcategory=?)"
            p += [cat, sub]
        for kind, op, value in (rating_filters or []):
            col = 'stars' if kind == 'star' else 'hearts'
            if op == 'gte':
                q += f" AND f.{col} >= ?"
            else:
                q += f" AND f.{col} = ?"
            p.append(value)
        q += " ORDER BY f.file_type DESC, f.filename COLLATE NOCASE"
        return self.conn.execute(q, p).fetchall()

    def search_files_flat(self, name="", inc_tags=None, exc_tags=None,
                          rating_filters=None, file_type=None) -> List:
        """
        Library-wide file search — unlike search_files(), this ignores
        folder structure entirely (no parent_id scoping). Used by
        BulkTagAssignDialog, where the user is picking files to tag
        regardless of which folder they live in.
        """
        q = "SELECT DISTINCT f.* FROM files f WHERE f.is_deleted=0 AND f.file_type != 'folder'"
        p = []
        if file_type:
            q += " AND f.file_type=?"; p.append(file_type)
        if name.strip():
            q += " AND f.filename LIKE ?"; p.append(f"%{name.strip()}%")
        for cat, sub in (inc_tags or []):
            q += " AND EXISTS(SELECT 1 FROM tags t WHERE t.file_id=f.id AND t.category=? AND t.subcategory=?)"
            p += [cat, sub]
        for cat, sub in (exc_tags or []):
            q += " AND NOT EXISTS(SELECT 1 FROM tags t WHERE t.file_id=f.id AND t.category=? AND t.subcategory=?)"
            p += [cat, sub]
        for kind, op, value in (rating_filters or []):
            col = 'stars' if kind == 'star' else 'hearts'
            if op == 'gte': q += f" AND f.{col} >= ?"
            else: q += f" AND f.{col} = ?"
            p.append(value)
        q += " ORDER BY f.filename COLLATE NOCASE LIMIT 2000"
        return self.conn.execute(q, p).fetchall()

    # ── Tag export/import (CSV backup) ──────────────────────────────────────

    def get_export_rows(self) -> List[dict]:
        """
        One row per (file, tag) pair for the full-assignment CSV export.
        Files with no tags at all still get one row (empty category/
        subcategory) so the complete file list is visible/editable.
        'status' is 'MISSING' if the file's abs_path no longer exists on
        disk at export time, else 'ok'.
        """
        files = self.conn.execute(
            "SELECT id,rel_path,filename,abs_path FROM files "
            "WHERE is_deleted=0 AND file_type != 'folder' "
            "ORDER BY rel_path COLLATE NOCASE").fetchall()
        rows = []
        for f in files:
            status = 'ok' if (f['abs_path'] and os.path.exists(f['abs_path'])) else 'MISSING'
            tags = self.get_tags(f['id'])
            if not tags:
                rows.append({'rel_path': f['rel_path'], 'filename': f['filename'],
                            'category': '', 'subcategory': '', 'status': status})
            else:
                for cat, sub in tags:
                    rows.append({'rel_path': f['rel_path'], 'filename': f['filename'],
                                'category': cat, 'subcategory': sub, 'status': status})
        return rows

    def get_vocab_export_rows(self) -> List[dict]:
        """Vocabulary-only export: just the category/subcategory structure,
        no file associations — for backing up/editing the tag taxonomy
        itself without the noise of every file assignment."""
        return [{'category': c, 'subcategory': s}
                for c, s in self.get_all_tag_categories()]

    def import_tag_rows(self, rows: List[dict]) -> Tuple[int, int, List[str]]:
        """
        Re-applies rows from get_export_rows()'s format. For each row with
        a non-empty category/subcategory: finds the file by rel_path and
        assigns the tag (additive — never removes existing tags, matching
        Bulk Assign's semantics). Rows whose rel_path doesn't match any
        file in this library are collected and returned, not applied.
        Returns (tags_applied, files_not_found_count, not_found_rel_paths).
        """
        rel_to_id = {r['rel_path']: r['id'] for r in self.conn.execute(
            "SELECT id, rel_path FROM files WHERE is_deleted=0")}
        applied = 0
        not_found: List[str] = []
        seen_missing: set = set()
        for row in rows:
            rel_path = (row.get('rel_path') or '').strip()
            cat = (row.get('category') or '').strip()
            sub = (row.get('subcategory') or '').strip()
            if not rel_path or not cat or not sub:
                continue   # blank tag rows (untagged files) carry no work to do
            fid = rel_to_id.get(rel_path)
            if fid is None:
                if rel_path not in seen_missing:
                    not_found.append(rel_path)
                    seen_missing.add(rel_path)
                continue
            self.bulk_assign_tag([fid], cat, sub)
            applied += 1
        return applied, len(not_found), not_found

    def import_vocab_rows(self, rows: List[dict]) -> int:
        """Re-applies rows from get_vocab_export_rows()'s format — adds
        each (category, subcategory) to the vocabulary. Returns how many
        were newly added (duplicates skipped)."""
        existing = set(self.get_all_tag_categories())
        added = 0
        for row in rows:
            cat = (row.get('category') or '').strip()
            sub = (row.get('subcategory') or '').strip()
            if not cat or not sub:
                continue
            if (cat, sub) not in existing:
                self.add_tag_to_vocab(cat, sub)
                existing.add((cat, sub))
                added += 1
        return added

    def add_file(self, filename, rel_path, abs_path, file_type,
                 file_size=None, checksum=None, width=None, height=None,
                 duration=None, date_modified=None, parent_id=None,
                 item_count=None) -> int:
        now = datetime.now().isoformat()
        cur = self.conn.execute("""
            INSERT OR REPLACE INTO files
            (filename,rel_path,abs_path,file_type,file_size,checksum,
             width,height,duration,date_added,date_modified,is_deleted,parent_id,
             item_count)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,0,?,?)""",
            (filename,rel_path,abs_path,file_type,file_size,checksum,
             width,height,duration,now,date_modified,parent_id,item_count))
        self.conn.commit(); return cur.lastrowid

    def set_item_count(self, fid: int, item_count: Optional[int]):
        self.conn.execute(
            "UPDATE files SET item_count=? WHERE id=?", (item_count, fid))
        self.conn.commit()

    def mark_deleted(self, fid: int):
        self.conn.execute("UPDATE files SET is_deleted=1 WHERE id=?", (fid,))
        self.conn.commit()

    def add_thumbnail(self, fid: int, frame: int, data: bytes):
        self.conn.execute(
            "INSERT OR REPLACE INTO thumbnails(file_id,frame_index,data) VALUES(?,?,?)",
            (fid, frame, data))
        self.conn.commit()

    def get_file_by_id(self, fid: int):
        return self.conn.execute("SELECT * FROM files WHERE id=?", (fid,)).fetchone()

    def get_tags(self, fid: int) -> List[Tuple[str,str]]:
        rows = self.conn.execute(
            "SELECT category,subcategory FROM tags WHERE file_id=? ORDER BY category,subcategory",
            (fid,)).fetchall()
        return [(r[0],r[1]) for r in rows]

    def set_tags(self, fid: int, tags: List[Tuple[str,str]]):
        self.conn.execute("DELETE FROM tags WHERE file_id=?", (fid,))
        for cat,sub in tags:
            cat, sub = cat.strip(), sub.strip()
            self.conn.execute(
                "INSERT INTO tags(file_id,category,subcategory) VALUES(?,?,?)",
                (fid,cat,sub))
            # Keep the vocabulary in sync — a tag typed fresh in the
            # per-item editor should also be visible in the Tag Manager.
            self.conn.execute(
                "INSERT OR IGNORE INTO tag_vocab(category,subcategory) VALUES(?,?)",
                (cat, sub))
        self.conn.commit()

    def get_all_tag_categories(self) -> List[Tuple[str,str]]:
        """All tags in the VOCABULARY (not just ones currently assigned to
        a file) — this is what makes 'create the tag first, assign later'
        possible. Falls back to nothing extra needed: tag_vocab is kept in
        sync with tags via add_tag_to_vocab / the migration backfill."""
        return [(r[0],r[1]) for r in self.conn.execute(
            "SELECT category,subcategory FROM tag_vocab ORDER BY category,subcategory")]

    # ── Tag vocabulary management (Tag Manager window) ──────────────────────

    def get_all_categories(self) -> List[str]:
        """Distinct category names only, for the category list panel."""
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT category FROM tag_vocab ORDER BY category")]

    def get_subcategories(self, category: str) -> List[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT subcategory FROM tag_vocab WHERE category=? ORDER BY subcategory",
            (category,))]

    def add_tag_to_vocab(self, category: str, subcategory: str):
        """Create a vocabulary entry without assigning it to any file yet."""
        cat, sub = category.strip(), subcategory.strip()
        if not cat or not sub: return
        self.conn.execute(
            "INSERT OR IGNORE INTO tag_vocab(category,subcategory) VALUES(?,?)",
            (cat, sub))
        self.conn.commit()

    def add_subcategories_bulk(self, category: str, subcategories: List[str]) -> int:
        """Add many subcategories under one category at once (the 'vehicle
        -> cars, trains, planes, boats' bulk-build workflow). Returns how
        many were newly added (duplicates are silently skipped)."""
        cat = category.strip()
        if not cat: return 0
        added = 0
        for sub in subcategories:
            sub = sub.strip()
            if not sub: continue
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO tag_vocab(category,subcategory) VALUES(?,?)",
                (cat, sub))
            if cur.rowcount: added += 1
        self.conn.commit()
        return added

    def rename_category(self, old_category: str, new_category: str):
        """Rename a category everywhere — vocabulary and every file
        assignment that uses it — in one transaction."""
        old, new = old_category.strip(), new_category.strip()
        if not old or not new or old == new: return
        self.conn.execute(
            "UPDATE tags SET category=? WHERE category=?", (new, old))
        # Vocab uses UNIQUE(category,subcategory); if the rename would
        # collide with an existing (new_category, subcategory) row, keep
        # the existing one and drop the old to avoid a constraint error.
        self.conn.execute("""
            DELETE FROM tag_vocab WHERE category=? AND subcategory IN
            (SELECT subcategory FROM tag_vocab WHERE category=?)
        """, (old, new))
        self.conn.execute(
            "UPDATE tag_vocab SET category=? WHERE category=?", (new, old))
        self.conn.commit()

    def rename_subcategory(self, category: str, old_sub: str, new_sub: str):
        """Rename a subcategory within one category, everywhere it's used."""
        cat, old, new = category.strip(), old_sub.strip(), new_sub.strip()
        if not cat or not old or not new or old == new: return
        self.conn.execute(
            "UPDATE tags SET subcategory=? WHERE category=? AND subcategory=?",
            (new, cat, old))
        self.conn.execute(
            "DELETE FROM tag_vocab WHERE category=? AND subcategory=?",
            (cat, new))  # drop pre-existing target to avoid UNIQUE collision
        self.conn.execute(
            "UPDATE tag_vocab SET subcategory=? WHERE category=? AND subcategory=?",
            (new, cat, old))
        self.conn.commit()

    def merge_tags(self, from_cat: str, from_sub: str, into_cat: str, into_sub: str):
        """Merge one tag into another: every file tagged with (from_cat,
        from_sub) gets (into_cat, into_sub) instead, then the old tag is
        removed from both assignments and the vocabulary."""
        fc,fs,ic,isub = from_cat.strip(),from_sub.strip(),into_cat.strip(),into_sub.strip()
        if not all([fc,fs,ic,isub]) or (fc,fs)==(ic,isub): return
        # Files that already have the target tag: just drop the source tag
        self.conn.execute("""
            DELETE FROM tags WHERE category=? AND subcategory=? AND file_id IN
            (SELECT file_id FROM tags WHERE category=? AND subcategory=?)
        """, (fc, fs, ic, isub))
        # Remaining files with the source tag: repoint to the target tag
        self.conn.execute(
            "UPDATE tags SET category=?,subcategory=? WHERE category=? AND subcategory=?",
            (ic, isub, fc, fs))
        self.conn.execute(
            "DELETE FROM tag_vocab WHERE category=? AND subcategory=?", (fc, fs))
        self.conn.execute(
            "INSERT OR IGNORE INTO tag_vocab(category,subcategory) VALUES(?,?)",
            (ic, isub))
        self.conn.commit()

    def delete_tag_globally(self, category: str, subcategory: str):
        """Remove a tag from the vocabulary AND every file it's assigned to."""
        cat, sub = category.strip(), subcategory.strip()
        self.conn.execute(
            "DELETE FROM tags WHERE category=? AND subcategory=?", (cat, sub))
        self.conn.execute(
            "DELETE FROM tag_vocab WHERE category=? AND subcategory=?", (cat, sub))
        self.conn.commit()

    def delete_category_globally(self, category: str):
        """Remove an entire category — all its subcategories — from the
        vocabulary and every file assignment."""
        cat = category.strip()
        self.conn.execute("DELETE FROM tags WHERE category=?", (cat,))
        self.conn.execute("DELETE FROM tag_vocab WHERE category=?", (cat,))
        self.conn.commit()

    def bulk_assign_tag(self, file_ids: List[int], category: str, subcategory: str):
        """Assign one tag to many files at once (additive — doesn't touch
        each file's other tags). Also ensures the tag exists in vocab."""
        cat, sub = category.strip(), subcategory.strip()
        if not cat or not sub or not file_ids: return
        self.conn.execute(
            "INSERT OR IGNORE INTO tag_vocab(category,subcategory) VALUES(?,?)",
            (cat, sub))
        for fid in file_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO tags(file_id,category,subcategory) VALUES(?,?,?)",
                (fid, cat, sub))
        self.conn.commit()

    def get_tag_usage_count(self, category: str, subcategory: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM tags WHERE category=? AND subcategory=?",
            (category.strip(), subcategory.strip())).fetchone()[0]

    # ── Ratings ──────────────────────────────────────────────────────────────

    def set_stars(self, fid: int, value: int):
        value = max(0, min(5, value))
        self.conn.execute("UPDATE files SET stars=? WHERE id=?", (value, fid))
        self.conn.commit()

    def set_hearts(self, fid: int, value: int):
        value = max(0, min(5, value))
        self.conn.execute("UPDATE files SET hearts=? WHERE id=?", (value, fid))
        self.conn.commit()

    def reset_ratings(self, fid: int):
        self.conn.execute("UPDATE files SET stars=0,hearts=0 WHERE id=?", (fid,))
        self.conn.commit()

    # ── Sources (tracked top-level import folders, for rescan) ─────────────────

    def add_source(self, abs_path: str, recursive: bool):
        self.conn.execute(
            "INSERT OR REPLACE INTO sources(abs_path,recursive,date_added) "
            "VALUES(?,?,COALESCE((SELECT date_added FROM sources WHERE abs_path=?),?))",
            (abs_path, 1 if recursive else 0, abs_path, datetime.now().isoformat()))
        self.conn.commit()

    def get_sources(self) -> List:
        return self.conn.execute("SELECT * FROM sources ORDER BY date_added").fetchall()

    # ── Compact (permanent purge of deleted rows + VACUUM) ──────────────────

    def compact(self) -> Tuple[int,int]:
        """
        Permanently remove rows marked is_deleted=1 (and their tags/thumbnails
        via ON DELETE CASCADE), then VACUUM to reclaim disk space.
        Returns (files_purged, bytes_reclaimed_estimate).
        """
        before_size = os.path.getsize(self.lib_path) if os.path.exists(self.lib_path) else 0
        purged = self.conn.execute(
            "SELECT COUNT(*) FROM files WHERE is_deleted=1").fetchone()[0]
        self.conn.execute("DELETE FROM files WHERE is_deleted=1")
        self.conn.commit()
        self.conn.execute("VACUUM")
        after_size = os.path.getsize(self.lib_path) if os.path.exists(self.lib_path) else 0
        return purged, max(0, before_size - after_size)

    def close(self):
        if getattr(self._local,'conn',None):
            self._local.conn.close(); self._local.conn = None

# ── Import worker ─────────────────────────────────────────────────────────────

class ImportWorker(QThread):
    progress     = pyqtSignal(int, int, str)
    file_done    = pyqtSignal(int)
    finished_sig = pyqtSignal(int, int)
    error_sig    = pyqtSignal(str)

    def __init__(self, db: LibraryDB, paths: List[str],
                 base_dir: str, recursive=False, parent_id=None):
        super().__init__()
        self.db=db; self.paths=paths; self.base_dir=base_dir
        self.recursive=recursive; self.parent_id=parent_id; self._cancel=False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            todo = []
            for p in self.paths:
                if os.path.isdir(p): self._collect(p, todo, is_root=True)
                elif Path(p).suffix.lower() in ALL_EXTS:
                    todo.append((p, self.parent_id))
            total=len(todo); added=skipped=0
            log.info("Import: %d files", total)
            for idx,(path,par) in enumerate(todo):
                if self._cancel: break
                fname = os.path.basename(path)
                self.progress.emit(idx+1, total, fname)
                try:
                    fid = self._import_one(path, par)
                    if fid: added+=1; self.file_done.emit(fid)
                    else: skipped+=1
                except Exception as e:
                    log.exception("import error %s", path)
                    self.error_sig.emit(f"{fname}: {e}"); skipped+=1
            self.finished_sig.emit(added, skipped)
        except Exception as e:
            log.exception("ImportWorker crashed")
            self.error_sig.emit(str(e)); self.finished_sig.emit(0,0)

    def _rel(self, path): 
        try: return os.path.relpath(path, self.base_dir)
        except ValueError: return path

    def _collect(self, dirpath, out, parent_id=None, is_root=False):
        """
        Walk dirpath and collect (file_path, parent_id) pairs into out.

        is_root=True means dirpath is the folder the user picked in the
        dialog — its files attach directly to self.parent_id (the folder
        the app was browsing when Add Folder was clicked), and dirpath
        itself does NOT get a folder row.

        is_root=False means dirpath is a subfolder found during a
        recursive scan — it DOES get its own 'folder' row in the DB
        (so it's navigable), and its contents attach to that row. If the
        folder directly contains images, a 2x2 mosaic of up to 4 of them
        (natural-sorted) is generated as the folder's thumbnail instead
        of the generic folder icon.
        """
        try: entries = sorted(os.scandir(dirpath), key=lambda e: e.name.lower())
        except PermissionError: return

        if is_root:
            par = self.parent_id
        elif self.recursive:
            rel = self._rel(dirpath)
            row = self.db.conn.execute(
                "SELECT id, item_count FROM files WHERE rel_path=?", (rel,)).fetchone()
            if row:
                par = row[0]
                # Backfill: existing folder row from before Focus view
                # existed. It may already have a frame-0 mosaic thumbnail
                # from the earlier mosaic-only feature — checking "any
                # thumbnail present" would wrongly treat that as already
                # done. item_count being NULL is the reliable signal that
                # Focus view's data (item_count + frames 1-7) is missing.
                if row['item_count'] is None:
                    self._generate_folder_thumbnail(par, dirpath, entries)
            else:
                par = self.db.add_file(
                    os.path.basename(dirpath), rel, dirpath, 'folder',
                    parent_id=parent_id)
                self._generate_folder_thumbnail(par, dirpath, entries)
        else:
            par = parent_id

        for e in entries:
            if self._cancel: return
            if e.is_dir(follow_symlinks=False):
                if self.recursive: self._collect(e.path, out, par, is_root=False)
            elif Path(e.name).suffix.lower() in ALL_EXTS:
                out.append((e.path, par))

    def _generate_folder_thumbnail(self, folder_id: int, dirpath, entries):
        """
        Build and store the folder's cover thumbnail: a 2x2 mosaic (up to
        4 images), used as the normal gallery view's single thumbnail.
        Also computes and stores item_count: every direct child (file or
        subfolder) counts as ONE item; a subfolder's own contents are not
        counted separately (matches the zip counting rule).
        """
        image_names = sorted(
            (e.name for e in entries
             if not e.is_dir(follow_symlinks=False)
             and Path(e.name).suffix.lower() in IMAGE_EXTS),
            key=natural_sort_key)

        item_count = sum(1 for _ in entries)   # every direct child, file or folder
        self.db.set_item_count(folder_id, item_count)

        if not image_names:
            return
        image_paths = [os.path.join(dirpath, n) for n in image_names]

        mosaic_data = thumb_folder_mosaic(image_paths[:4])
        if mosaic_data:
            self.db.add_thumbnail(folder_id, 0, mosaic_data)

    def _import_one(self, path, parent_id) -> Optional[int]:
        ext = Path(path).suffix.lower()
        rel = self._rel(path)
        row = self.db.conn.execute(
            "SELECT id,is_deleted FROM files WHERE rel_path=?", (rel,)).fetchone()
        if row and not row['is_deleted']: return None
        stat = os.stat(path)
        checksum = md5_file(path)
        ftype = ('image' if ext in IMAGE_EXTS else
                 'video' if ext in VIDEO_EXTS else 'zip')
        w=h=dur=None; thumbs=[]; item_count=None
        if ftype=='image':
            try:
                with Image.open(path) as img:
                    w,h=img.size; b=_pil_to_jpeg(img)
                    if b: thumbs=[b]
            except Exception: pass
        elif ftype=='video':
            try:
                cap=cv2.VideoCapture(path)
                if cap.isOpened():
                    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps=cap.get(cv2.CAP_PROP_FPS) or 1
                    dur=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
                cap.release()
                thumbs=thumb_video(path)
            except Exception: pass
        elif ftype=='zip':
            thumbs, item_count = analyze_zip_contents(path)
        dm = datetime.fromtimestamp(stat.st_mtime).isoformat()
        if row:
            fid=row['id']
            self.db.conn.execute("""UPDATE files SET filename=?,abs_path=?,
                file_size=?,checksum=?,width=?,height=?,duration=?,
                date_modified=?,is_deleted=0,parent_id=?,item_count=? WHERE id=?""",
                (os.path.basename(path),path,stat.st_size,checksum,
                 w,h,dur,dm,parent_id,item_count,fid))
            self.db.conn.execute("DELETE FROM thumbnails WHERE file_id=?",(fid,))
            self.db.conn.commit()
        else:
            fid=self.db.add_file(os.path.basename(path),rel,path,ftype,
                stat.st_size,checksum,w,h,dur,dm,parent_id,item_count)
        for i,b in enumerate(thumbs): self.db.add_thumbnail(fid,i,b)
        return fid

# ── Health-check worker ───────────────────────────────────────────────────────

class HealthCheckWorker(QThread):
    progress     = pyqtSignal(int, int, str)
    finished_sig = pyqtSignal(int)
    error_sig    = pyqtSignal(str)

    def __init__(self, lib_path: str):
        super().__init__(); self.lib_path=lib_path; self._cancel=False

    def cancel(self): self._cancel=True

    def run(self):
        try:
            conn=sqlite3.connect(self.lib_path,timeout=30)
            conn.row_factory=sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            rows=conn.execute(
                "SELECT id,abs_path,filename FROM files WHERE is_deleted=0").fetchall()
            total=len(rows); issues=0
            log.debug("HealthCheck: %d files", total)
            for i,row in enumerate(rows):
                if self._cancel: break
                self.progress.emit(i+1,total,row['filename'])
                if row['abs_path'] and not os.path.exists(row['abs_path']):
                    conn.execute("UPDATE files SET is_deleted=1 WHERE id=?",(row['id'],))
                    conn.commit(); issues+=1
            conn.close()
            log.debug("HealthCheck done: %d issues", issues)
            self.finished_sig.emit(issues)
        except Exception as e:
            log.exception("HealthCheckWorker crashed")
            self.error_sig.emit(str(e)); self.finished_sig.emit(0)

# ── Rescan worker ──────────────────────────────────────────────────────────────

class RescanWorker(QThread):
    """
    Rescans every tracked source folder for the current library:
      - New files on disk but not in DB      -> imported (added='is_deleted' aware)
      - Files in DB but missing on disk       -> marked deleted (is_deleted=1)
      - Files back on disk after being marked
        deleted                               -> restored (tags kept, is_deleted=0)
      - Files whose MD5 checksum changed      -> metadata + thumbnails regenerated
      - A file missing from its old rel_path AND a new file elsewhere with
        the SAME MD5 checksum -> treated as a rename/move: the existing
        DB row is re-pointed to the new location (rel_path/abs_path/
        filename/parent_id updated), so its tags, ratings, and thumbnails
        carry over instead of being lost as a delete+fresh-import.
        Matching is strictly one-to-one; if a checksum is ambiguous (more
        than one candidate on either side) it's left alone and falls back
        to the normal add/delete behavior rather than risk relinking the
        wrong file.
    Uses its own SQLite connection — never shares the main thread's.
    """
    progress     = pyqtSignal(int, int, str)
    finished_sig = pyqtSignal(int, int, int, int, int)  # added, restored, removed, updated, relinked
    error_sig    = pyqtSignal(str)

    def __init__(self, lib_path: str, base_dir: str):
        super().__init__()
        self.lib_path = lib_path
        self.base_dir = base_dir
        self._cancel  = False

    def cancel(self): self._cancel = True

    def _rel(self, path):
        try: return os.path.relpath(path, self.base_dir)
        except ValueError: return path

    def run(self):
        try:
            conn = sqlite3.connect(self.lib_path, timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")

            sources = conn.execute("SELECT * FROM sources").fetchall()
            if not sources:
                conn.close()
                self.finished_sig.emit(0,0,0,0,0)
                return

            # ── Step 1: walk every source folder, collect files on disk.
            # Mirrors ImportWorker._collect's folder-hierarchy logic exactly:
            # each subfolder gets/reuses a 'folder' row, and every file's
            # parent_id is set to its containing folder's row id (or the
            # library root, NULL, for files directly in the source folder).
            on_disk = {}   # rel_path -> (abs_path, parent_id)
            for src in sources:
                self._walk(conn, src['abs_path'], bool(src['recursive']),
                          on_disk, parent_id=None, is_root=True)
                if self._cancel: break

            # ── Step 2: compare against DB ──────────────────────────────────────
            db_rows = conn.execute(
                "SELECT id,rel_path,abs_path,checksum,is_deleted,file_type,"
                "filename,parent_id,file_size,date_modified,item_count FROM files "
                "WHERE file_type != 'folder'").fetchall()
            db_by_rel = {r['rel_path']: r for r in db_rows}

            # ── Step 2.5: rename/move detection via checksum matching ──────────
            # Candidates for relinking: paths that are new-on-disk (not in
            # db_by_rel) matched against DB rows whose rel_path is missing
            # from disk. Only genuinely "missing" rows are eligible — a row
            # that's already marked is_deleted stays as a normal restore
            # candidate in Step 3, not a relink target, since restoring it
            # in place (same rel_path reappearing) is simpler and correct
            # as-is.
            new_paths = {rp: ap for rp, (ap, _pid) in on_disk.items()
                        if rp not in db_by_rel}
            missing_rows = {r['rel_path']: r for r in db_rows
                           if r['rel_path'] not in on_disk and not r['is_deleted']}

            relinked = 0
            relinked_new_paths: set = set()     # new_paths keys consumed by a relink
            relinked_missing_rels: set = set()  # missing_rows keys consumed by a relink

            if new_paths and missing_rows:
                # Checksum every candidate on both sides. New-file hashing
                # is unavoidable (needed either way, for relink or fresh
                # import); missing files already have their checksum
                # stored from before, so no disk read is needed for them.
                relink_total = len(new_paths)
                relink_step  = 0
                new_checksums: Dict[str, List[str]] = {}   # checksum -> [rel_path,...]
                for rp, ap in new_paths.items():
                    if self._cancel: break
                    relink_step += 1
                    self.progress.emit(relink_step, relink_total,
                                       f"Checking for renamed/moved files: {os.path.basename(ap)}")
                    cs = md5_file(ap)
                    if cs:
                        new_checksums.setdefault(cs, []).append(rp)

                missing_checksums: Dict[str, List[str]] = {}
                for rp, row in missing_rows.items():
                    if row['checksum']:
                        missing_checksums.setdefault(row['checksum'], []).append(rp)

                for checksum, missing_rels in missing_checksums.items():
                    if self._cancel: break
                    candidate_new_rels = new_checksums.get(checksum)
                    if not candidate_new_rels:
                        continue
                    # Strictly one-to-one only — ambiguous on either side
                    # is left for the normal add/delete path instead of
                    # guessing which pair actually corresponds.
                    if len(missing_rels) != 1 or len(candidate_new_rels) != 1:
                        continue
                    old_rel = missing_rels[0]
                    new_rel = candidate_new_rels[0]
                    old_row = missing_rows[old_rel]
                    new_abs_path, new_parent_id = on_disk[new_rel]

                    conn.execute("""
                        UPDATE files SET rel_path=?,abs_path=?,filename=?,
                        parent_id=?,is_deleted=0 WHERE id=?""",
                        (new_rel, new_abs_path, os.path.basename(new_abs_path),
                         new_parent_id, old_row['id']))
                    conn.commit()
                    relinked += 1
                    relinked_new_paths.add(new_rel)
                    relinked_missing_rels.add(old_rel)

            total = len(on_disk) + len(db_rows)
            step  = 0
            added = restored = removed = updated = 0

            # ── Step 3: files present on disk ───────────────────────────────────
            for rel_path, (abs_path, parent_id) in on_disk.items():
                if self._cancel: break
                if rel_path in relinked_new_paths:
                    continue   # already handled in Step 2.5 — don't double-import
                step += 1
                self.progress.emit(step, total, os.path.basename(abs_path))

                existing = db_by_rel.get(rel_path)
                if existing is None:
                    # Brand new file — import it fully, correctly parented
                    fid = self._import_new(conn, abs_path, rel_path, parent_id)
                    if fid: added += 1
                    continue

                if existing['is_deleted']:
                    # Was removed, now back — restore, keep tags
                    conn.execute("UPDATE files SET is_deleted=0,abs_path=? WHERE id=?",
                                 (abs_path, existing['id']))
                    conn.commit()
                    restored += 1
                    # Fall through: still check checksum below in case content changed too

                # Repair parent_id if it doesn't match the current on-disk
                # location — heals files that were previously imported with
                # broken folder linkage (e.g. from an earlier buggy rescan).
                if existing['parent_id'] != parent_id:
                    conn.execute("UPDATE files SET parent_id=? WHERE id=?",
                                 (parent_id, existing['id']))
                    conn.commit()

                # Change detection: os.stat() (cheap) first — only fall
                # back to a full MD5 read (expensive: reads the entire
                # file) when mtime or size actually differ from what's
                # stored. At 50k-100k files, hashing everything on every
                # rescan regardless of whether it changed would make
                # rescan prohibitively slow; this keeps it fast for the
                # overwhelming common case of "nothing changed".
                try:
                    st = os.stat(abs_path)
                    stat_changed = (
                        existing['file_size'] != st.st_size or
                        existing['date_modified'] != datetime.fromtimestamp(st.st_mtime).isoformat()
                    )
                except OSError:
                    stat_changed = True   # can't stat — be safe, check fully

                if stat_changed:
                    new_checksum = md5_file(abs_path)
                    if new_checksum and new_checksum != existing['checksum']:
                        self._regenerate(conn, existing['id'], abs_path,
                                        existing['file_type'], new_checksum)
                        updated += 1
                elif existing['file_type'] == 'zip' and existing['item_count'] is None:
                    # Backfill: this zip's content hasn't changed (so the
                    # normal update path above never ran), but it predates
                    # the multi-thumbnail/item_count feature — its
                    # item_count column is still NULL. Regenerate now so
                    # existing libraries get Focus view thumbnails and
                    # counts without needing every file to "change" first.
                    self._regenerate(conn, existing['id'], abs_path,
                                    existing['file_type'], existing['checksum'])
                    updated += 1

            # ── Step 4: files in DB but missing on disk ─────────────────────────
            for rel_path, row in db_by_rel.items():
                if self._cancel: break
                if rel_path in relinked_missing_rels:
                    continue   # already handled in Step 2.5 — don't mark deleted
                step += 1
                if rel_path not in on_disk and not row['is_deleted']:
                    conn.execute("UPDATE files SET is_deleted=1 WHERE id=?", (row['id'],))
                    conn.commit()
                    removed += 1
                self.progress.emit(step, total, row['filename'])

            conn.close()
            log.info("Rescan done: +%d added, %d restored, %d removed, %d updated, %d relinked",
                     added, restored, removed, updated, relinked)
            self.finished_sig.emit(added, restored, removed, updated, relinked)

        except Exception as e:
            log.exception("RescanWorker crashed")
            self.error_sig.emit(str(e))
            self.finished_sig.emit(0,0,0,0,0)

    def _walk(self, conn, dirpath: str, recursive: bool, out: dict,
              parent_id=None, is_root=False):
        """
        Populate out[rel_path] = (abs_path, parent_id) for every media file
        under dirpath. Mirrors ImportWorker._collect exactly:
          is_root=True  -> dirpath is a tracked source folder itself; its
                            files attach to `parent_id` (the folder the
                            source was originally imported into — normally
                            the library root, NULL), and dirpath does NOT
                            get its own folder row.
          is_root=False -> dirpath is a subfolder found during a recursive
                            walk; it DOES get/reuse a 'folder' row, and its
                            contents attach to that row's id. A newly
                            created folder row gets a 2x2 mosaic thumbnail
                            from up to 4 of its direct-child images.
        """
        if not os.path.isdir(dirpath):
            return

        try:
            entries = sorted(os.scandir(dirpath), key=lambda e: e.name.lower())
        except PermissionError:
            return

        if is_root:
            par = parent_id
        elif recursive:
            rel = self._rel(dirpath)
            row = conn.execute(
                "SELECT id, item_count FROM files WHERE rel_path=?", (rel,)).fetchone()
            if row:
                par = row[0]
                # Backfill: this folder row already existed. It may already
                # have a frame-0 mosaic thumbnail from before Focus view
                # existed — checking "any thumbnail present" would wrongly
                # skip it, since that's exactly the folder's pre-existing
                # state. The reliable signal that Focus view's data
                # (item_count + frames 1-7) hasn't been generated yet is
                # item_count itself being NULL, so check that instead.
                if row['item_count'] is None:
                    self._generate_folder_thumbnail(conn, par, dirpath, entries)
            else:
                now = datetime.now().isoformat()
                cur = conn.execute("""
                    INSERT INTO files
                    (filename,rel_path,abs_path,file_type,date_added,is_deleted,parent_id)
                    VALUES(?,?,?,?,?,0,?)""",
                    (os.path.basename(dirpath), rel, dirpath, 'folder', now, parent_id))
                conn.commit()
                par = cur.lastrowid
                self._generate_folder_thumbnail(conn, par, dirpath, entries)
        else:
            par = parent_id

        for e in entries:
            if self._cancel: return
            if e.is_dir(follow_symlinks=False):
                if recursive:
                    self._walk(conn, e.path, True, out, parent_id=par, is_root=False)
            elif Path(e.name).suffix.lower() in ALL_EXTS:
                out[self._rel(e.path)] = (e.path, par)

    def _generate_folder_thumbnail(self, conn, folder_id: int, dirpath, entries):
        """
        Build and store the folder's cover thumbnail (2x2 mosaic, frame 0)
        — same as ImportWorker._generate_folder_thumbnail. Also
        (re)computes item_count. Uses the caller's own connection
        (RescanWorker's thread-local one)."""
        image_names = sorted(
            (e.name for e in entries
             if not e.is_dir(follow_symlinks=False)
             and Path(e.name).suffix.lower() in IMAGE_EXTS),
            key=natural_sort_key)

        item_count = sum(1 for _ in entries)
        conn.execute("UPDATE files SET item_count=? WHERE id=?", (item_count, folder_id))
        conn.commit()

        if not image_names:
            return
        image_paths = [os.path.join(dirpath, n) for n in image_names]

        mosaic_data = thumb_folder_mosaic(image_paths[:4])
        if mosaic_data:
            conn.execute(
                "INSERT OR REPLACE INTO thumbnails(file_id,frame_index,data) VALUES(?,0,?)",
                (folder_id, mosaic_data))
        conn.commit()

    def _import_new(self, conn, path: str, rel_path: str, parent_id) -> Optional[int]:
        """Import a file that's on disk but has no DB row at all yet."""
        ext = Path(path).suffix.lower()
        try:
            stat = os.stat(path)
        except OSError:
            return None
        checksum = md5_file(path)
        ftype = ('image' if ext in IMAGE_EXTS else
                 'video' if ext in VIDEO_EXTS else 'zip')
        w=h=dur=None; thumbs=[]; item_count=None
        if ftype=='image':
            try:
                with Image.open(path) as img:
                    w,h=img.size; b=_pil_to_jpeg(img)
                    if b: thumbs=[b]
            except Exception: pass
        elif ftype=='video':
            try:
                cap=cv2.VideoCapture(path)
                if cap.isOpened():
                    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps=cap.get(cv2.CAP_PROP_FPS) or 1
                    dur=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
                cap.release()
                thumbs=thumb_video(path)
            except Exception: pass
        elif ftype=='zip':
            thumbs, item_count = analyze_zip_contents(path)
        dm = datetime.fromtimestamp(stat.st_mtime).isoformat()
        now = datetime.now().isoformat()
        cur = conn.execute("""
            INSERT OR IGNORE INTO files
            (filename,rel_path,abs_path,file_type,file_size,checksum,
             width,height,duration,date_added,date_modified,is_deleted,parent_id,
             item_count)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,0,?,?)""",
            (os.path.basename(path),rel_path,path,ftype,stat.st_size,checksum,
             w,h,dur,now,dm,parent_id,item_count))
        conn.commit()
        fid = cur.lastrowid
        if not fid:
            return None
        for i,b in enumerate(thumbs):
            conn.execute(
                "INSERT OR REPLACE INTO thumbnails(file_id,frame_index,data) VALUES(?,?,?)",
                (fid,i,b))
        conn.commit()
        return fid

    def _regenerate(self, conn, fid: int, path: str, ftype: str, checksum: str = None):
        """Checksum changed — refresh metadata + thumbnails for an existing row.
        checksum is normally passed in already-computed by the caller (which
        needed it anyway to detect the change) to avoid reading the whole
        file a second time; falls back to computing it here if called
        without one."""
        try:
            stat = os.stat(path)
        except OSError:
            return
        if checksum is None:
            checksum = md5_file(path)
        w=h=dur=None; thumbs=[]; item_count=None
        if ftype=='image':
            try:
                with Image.open(path) as img:
                    w,h=img.size; b=_pil_to_jpeg(img)
                    if b: thumbs=[b]
            except Exception: pass
        elif ftype=='video':
            try:
                cap=cv2.VideoCapture(path)
                if cap.isOpened():
                    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps=cap.get(cv2.CAP_PROP_FPS) or 1
                    dur=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
                cap.release()
                thumbs=thumb_video(path)
            except Exception: pass
        elif ftype=='zip':
            thumbs, item_count = analyze_zip_contents(path)
        dm = datetime.fromtimestamp(stat.st_mtime).isoformat()
        conn.execute("""UPDATE files SET file_size=?,checksum=?,width=?,height=?,
            duration=?,date_modified=?,item_count=? WHERE id=?""",
            (stat.st_size,checksum,w,h,dur,dm,item_count,fid))
        conn.execute("DELETE FROM thumbnails WHERE file_id=?", (fid,))
        for i,b in enumerate(thumbs):
            conn.execute(
                "INSERT OR REPLACE INTO thumbnails(file_id,frame_index,data) VALUES(?,?,?)",
                (fid,i,b))
        conn.commit()

# ── Compact worker ─────────────────────────────────────────────────────────────

class CompactWorker(QThread):
    """Permanently purges deleted rows and VACUUMs the database file."""
    finished_sig = pyqtSignal(int, int)   # files_purged, bytes_reclaimed
    error_sig    = pyqtSignal(str)

    def __init__(self, lib_path: str):
        super().__init__(); self.lib_path = lib_path

    def run(self):
        try:
            conn = sqlite3.connect(self.lib_path, timeout=30)
            before = os.path.getsize(self.lib_path) if os.path.exists(self.lib_path) else 0
            purged = conn.execute(
                "SELECT COUNT(*) FROM files WHERE is_deleted=1").fetchone()[0]
            conn.execute("DELETE FROM files WHERE is_deleted=1")
            conn.commit()
            conn.execute("VACUUM")
            conn.close()
            after = os.path.getsize(self.lib_path) if os.path.exists(self.lib_path) else 0
            log.info("Compact done: purged=%d reclaimed=%d bytes", purged, max(0,before-after))
            self.finished_sig.emit(purged, max(0, before - after))
        except Exception as e:
            log.exception("CompactWorker crashed")
            self.error_sig.emit(str(e))
            self.finished_sig.emit(0, 0)

# ── Thumbnail fetch worker ────────────────────────────────────────────────────

class ThumbFetcher(QThread):
    """
    Fetches JPEG bytes from SQLite in a background thread.
    Decodes to QImage (thread-safe) and emits (file_id, QImage, frame_count).
    The main thread only does QPixmap.fromImage() — one line, no layout work.
    """
    result = pyqtSignal(int, object, int)   # file_id, QImage|None, frame_count

    BATCH = 30      # emit every N items so the signal queue stays small
    SLEEP = 8       # ms between batches

    def __init__(self, lib_path: str, ids: List[int]):
        super().__init__()
        self.lib_path=lib_path; self.ids=ids; self._cancel=False

    def cancel(self): self._cancel=True

    def run(self):
        try:
            conn=sqlite3.connect(self.lib_path,timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA mmap_size=268435456")
            for n,fid in enumerate(self.ids):
                if self._cancel: break
                row=conn.execute(
                    "SELECT data FROM thumbnails WHERE file_id=? AND frame_index=0",
                    (fid,)).fetchone()
                data = row[0] if row else None
                cnt=conn.execute(
                    "SELECT COUNT(*) FROM thumbnails WHERE file_id=?",
                    (fid,)).fetchone()[0]
                # Decode here — QImage is thread-safe, QPixmap is not
                qimg = jpeg_to_qimage(data) if data else None
                self.result.emit(fid, qimg, cnt)
                if (n+1) % self.BATCH == 0:
                    self.msleep(self.SLEEP)
            conn.close()
        except Exception:
            log.exception("ThumbFetcher crashed")


# ── Hover frame loader ────────────────────────────────────────────────────────

class HoverLoader(QThread):
    """
    Fetches all thumbnail frames for one video file.
    Decodes to QImage off-thread (thread-safe).
    Emits frames_ready(fid, list[QImage]) on the main thread via Qt signal.
    Never touches QPixmap — that happens in the connected slot on main thread.
    """
    frames_ready = pyqtSignal(int, list)   # fid, [QImage, ...]

    def __init__(self, lib_path: str, fid: int, cell_size: int):
        super().__init__()
        self.lib_path  = lib_path
        self.fid       = fid
        self.cell_size = cell_size

    def run(self):
        try:
            conn = sqlite3.connect(self.lib_path, timeout=10)
            rows = conn.execute(
                "SELECT data FROM thumbnails WHERE file_id=? ORDER BY frame_index",
                (self.fid,)).fetchall()
            conn.close()
            qimages = []
            for r in rows:
                qi = jpeg_to_qimage(r[0])
                if qi:
                    sc = qi.scaled(self.cell_size, self.cell_size,
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    qimages.append(sc)
            if qimages:
                self.frames_ready.emit(self.fid, qimages)
        except Exception:
            log.exception("HoverLoader failed fid=%d", self.fid)

# ── Zip content thumbnail loader (for PDF export preview) ────────────────────

class ZipThumbLoader(QThread):
    """
    Extracts every image inside a zip to the session temp directory and
    generates a preview-sized QImage for each, streamed progressively via
    thumb_ready so the preview dialog can populate incrementally instead
    of blocking on the whole zip. Natural-sorted by filename.

    Extracted full-resolution files are kept in the session temp dir
    (not deleted immediately) so the PDF export step can reuse them
    without re-extracting — the whole temp dir is wiped on app exit.
    """
    thumb_ready  = pyqtSignal(int, str, object)   # index, entry_name, QImage|None
    finished_sig = pyqtSignal(int)                # total image count
    error_sig    = pyqtSignal(str)

    PREVIEW_SIZE = 180   # px, for the preview grid thumbnail

    def __init__(self, zip_path: str, extract_dir: str):
        super().__init__()
        self.zip_path    = zip_path
        self.extract_dir = extract_dir
        self._cancel     = False

    def cancel(self): self._cancel = True

    def run(self):
        try:
            os.makedirs(self.extract_dir, exist_ok=True)
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                names = sorted(
                    (n for n in zf.namelist()
                     if Path(n).suffix.lower() in IMAGE_EXTS
                     and not n.endswith('/')),
                    key=natural_sort_key)

                for idx, name in enumerate(names):
                    if self._cancel: break
                    try:
                        data = zf.read(name)
                        # Extract to a flat filename in the temp dir so page
                        # ordering survives even if the zip has subfolders
                        out_name = f"{idx:05d}_{Path(name).name}"
                        out_path = os.path.join(self.extract_dir, out_name)
                        with open(out_path, 'wb') as f:
                            f.write(data)

                        img = Image.open(io.BytesIO(data))
                        img.thumbnail((self.PREVIEW_SIZE, self.PREVIEW_SIZE), Image.LANCZOS)
                        buf = io.BytesIO()
                        if img.mode not in ('RGB','L'): img = img.convert('RGB')
                        img.save(buf, format='JPEG', quality=80)
                        qimg = jpeg_to_qimage(buf.getvalue())
                        self.thumb_ready.emit(idx, out_path, qimg)
                    except Exception:
                        log.exception("ZipThumbLoader: failed on %s", name)
                        self.thumb_ready.emit(idx, "", None)

                self.finished_sig.emit(len(names))
        except Exception as e:
            log.exception("ZipThumbLoader crashed")
            self.error_sig.emit(str(e))
            self.finished_sig.emit(0)

# ── Reader page enhancement worker ────────────────────────────────────────────

class PageEnhanceWorker(QThread):
    """
    Pre-processes every page in a reader session with noise reduction,
    sharpen, and contrast, writing enhanced copies to the session temp
    dir. Used when the reader's A1 (mild) or A2 (strong) toggle is
    selected — per spec, this renders ALL pages upfront rather than
    lazily per page-turn, so flipping pages afterward is instant.
    """
    progress     = pyqtSignal(int, int, str)
    finished_sig = pyqtSignal(dict)   # {original_path: enhanced_path}
    error_sig    = pyqtSignal(str)

    def __init__(self, page_paths: List[str], sharpen_percent: float,
                 contrast_factor: float, out_dir: str, noise_percent: float = 0):
        super().__init__()
        self.page_paths      = page_paths
        self.sharpen_percent = sharpen_percent
        self.contrast_factor = contrast_factor
        self.noise_percent   = noise_percent
        self.out_dir         = out_dir
        self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        result = {}
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            total = len(self.page_paths)
            for i, path in enumerate(self.page_paths):
                if self._cancel: break
                self.progress.emit(i+1, total, os.path.basename(path))
                try:
                    with Image.open(path) as img:
                        if img.mode not in ('RGB','L'):
                            img = img.convert('RGB')
                        img = enhance_image(img, self.sharpen_percent,
                                           self.contrast_factor, self.noise_percent)
                        out_path = os.path.join(self.out_dir, os.path.basename(path))
                        img.save(out_path, format='JPEG', quality=92)
                        result[path] = out_path
                except Exception:
                    log.exception("PageEnhanceWorker: failed on %s", path)
                    result[path] = path   # fall back to original on failure
            self.finished_sig.emit(result)
        except Exception as e:
            log.exception("PageEnhanceWorker crashed")
            self.error_sig.emit(str(e))
            self.finished_sig.emit(result)

# ── PDF export worker ─────────────────────────────────────────────────────────

class PdfExportWorker(QThread):
    """
    Builds one PDF per zip. Two modes:
      - file_paths given directly (preview dialog already extracted them,
        in final page order) -> just assemble the PDF.
      - zip_path given instead -> extract natural-sorted images to a temp
        dir first, then assemble. Used for the "multiple zips selected,
        no preview" path.
    Each page is sized to match its source image's own dimensions.

    Optional sharpen/contrast/noise-reduction enhancement is applied
    per-page, relative to each image's own values (ImageEnhance/UnsharpMask
    work this way natively — a factor of 1.10 means 10% more than that
    image already has). Pass sharpen_percent=100, contrast_factor=1.0,
    noise_percent=0 (the defaults) to export unmodified — this matches
    enhance_params_for_level('off').
    """
    progress     = pyqtSignal(int, int, str)   # current, total, label
    file_done    = pyqtSignal(str, bool)       # output pdf path, success
    finished_sig = pyqtSignal(int, int)        # pdfs_created, pdfs_failed
    error_sig    = pyqtSignal(str)

    def __init__(self, jobs: List[dict], out_dir: str,
                 sharpen_percent: float = 100, contrast_factor: float = 1.0,
                 noise_percent: float = 0):
        """
        jobs: list of dicts, each either
          {'pdf_name': str, 'file_paths': [str, ...]}          (pre-extracted)
          {'pdf_name': str, 'zip_path': str}                    (extract first)
        out_dir: destination folder for the PDFs (never the library folder).
        """
        super().__init__()
        self.jobs    = jobs
        self.out_dir = out_dir
        self.sharpen_percent = sharpen_percent
        self.contrast_factor = contrast_factor
        self.noise_percent   = noise_percent
        self._cancel = False

    def cancel(self): self._cancel = True

    def run(self):
        created = failed = 0
        total_jobs = len(self.jobs)
        try:
            for job_idx, job in enumerate(self.jobs):
                if self._cancel: break
                pdf_name = job['pdf_name']
                out_path = os.path.join(self.out_dir, pdf_name)

                try:
                    if 'file_paths' in job:
                        paths = job['file_paths']
                    else:
                        paths = self._extract_zip(job['zip_path'])

                    if not paths:
                        self.file_done.emit(out_path, False)
                        failed += 1
                        continue

                    self._build_pdf(paths, out_path, job_idx, total_jobs, pdf_name)
                    self.file_done.emit(out_path, True)
                    created += 1
                except Exception:
                    log.exception("PDF export failed for %s", pdf_name)
                    self.file_done.emit(out_path, False)
                    failed += 1

            self.finished_sig.emit(created, failed)
        except Exception as e:
            log.exception("PdfExportWorker crashed")
            self.error_sig.emit(str(e))
            self.finished_sig.emit(created, failed)

    def _extract_zip(self, zip_path: str) -> List[str]:
        """Extract all images from a zip to a temp dir, natural-sorted."""
        extract_dir = os.path.join(get_session_tmp_dir(),
                                    f"export_{os.path.basename(zip_path)}_{id(self)}")
        os.makedirs(extract_dir, exist_ok=True)
        paths = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = sorted(
                (n for n in zf.namelist()
                 if Path(n).suffix.lower() in IMAGE_EXTS
                 and not n.endswith('/')),
                key=natural_sort_key)
            for idx, name in enumerate(names):
                if self._cancel: break
                data = zf.read(name)
                out_name = f"{idx:05d}_{Path(name).name}"
                out_path = os.path.join(extract_dir, out_name)
                with open(out_path, 'wb') as f:
                    f.write(data)
                paths.append(out_path)
        return paths

    def _build_pdf(self, image_paths: List[str], out_path: str,
                    job_idx: int, total_jobs: int, pdf_name: str):
        """One page per image, page size = that image's own pixel dimensions
        (reportlab uses points == pixels here, 1:1, so the PDF page matches
        the source image's native resolution exactly)."""
        c = None
        total_pages = len(image_paths)
        for i, path in enumerate(image_paths):
            if self._cancel: break
            # cur/tot = whole-PDF count (so ProgressDialog's own "[cur/tot]"
            # prefix reads sensibly, e.g. "[1/3] file.pdf — page 4/20");
            # per-page detail is appended in the label text.
            label = f"{pdf_name} — page {i+1}/{total_pages}"
            self.progress.emit(job_idx+1, total_jobs, label)
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    if img.mode not in ('RGB','L'):
                        img = img.convert('RGB')
                    img = enhance_image(img, self.sharpen_percent,
                                       self.contrast_factor, self.noise_percent)
                    if c is None:
                        c = pdf_canvas.Canvas(out_path, pagesize=(w, h))
                    else:
                        c.setPageSize((w, h))
                    c.drawImage(ImageReader(img), 0, 0, width=w, height=h)
                    c.showPage()
            except Exception:
                log.exception("PDF export: failed to add page for %s", path)
        if c is not None:
            c.save()


# ── Gallery canvas ────────────────────────────────────────────────────────────
#
# ONE widget, zero child widgets per item.
# All items are painted directly in paintEvent.
# Mouse events hit-test against a simple grid formula.
# Hover animation uses a single QTimer that calls update().
#

class GalleryCanvas(QAbstractScrollArea):
    """
    Painter-based virtual gallery.
    - No QWidget per item → no layout passes, no stylesheet parsing per card.
    - QImage decoded off-thread; QPixmap created on main thread only when needed.
    - Pixmap cache keyed by (file_id, cell_size) — evicted on zoom or view change.
    - Single QTimer drives hover animation for the hovered item only.
    """

    selection_changed = pyqtSignal(list)     # [file_id, ...]
    item_double_click = pyqtSignal(int)      # file_id
    item_right_click  = pyqtSignal(int)      # file_id
    navigate_folder   = pyqtSignal(int, str) # file_id, name
    zoom_changed      = pyqtSignal(int)      # new zoom %, so toolbar slider stays in sync

    def __init__(self, db: Optional[LibraryDB], config: AppConfig, parent=None):
        super().__init__(parent)
        self.db     = db
        self.config = config

        # View state
        self._rows: List        = []          # sqlite3.Row list
        self._scale: float      = float(config.get('thumb_scale','1.0'))
        self._cell_size: int    = self._calc_cell()
        self._parent_stack: List[Tuple[Optional[int],str]] = [(None,"Library")]

        # Pixmap cache: file_id → QPixmap (frame 0)
        self._pixmap_cache: Dict[int, QPixmap] = {}
        # Frame cache for hover: file_id → List[QPixmap]
        self._frame_cache: Dict[int, List[QPixmap]] = {}
        # Frame counts: file_id → int
        self._frame_counts: Dict[int, int] = {}

        # Selection
        self._selected: List[int]       = []
        self._last_click_idx: Optional[int] = None

        # Hover
        self._hover_idx: int            = -1   # row index of hovered item
        self._hover_frame: int          = 0
        self._hover_timer               = QTimer(self)
        self._hover_timer.setInterval(HOVER_MS)
        self._hover_timer.timeout.connect(self._advance_hover)
        self._hover_loading: set        = set()  # fids currently being fetched
        self._hover_loaders: Dict[int, 'HoverLoader'] = {}  # fid → active loader

        # Fetcher management — one active fetcher + pending queue
        self._fetchers: List[ThumbFetcher] = []
        self._fetch_queue: List[int]        = []   # ids waiting to be fetched
        self._fetch_queue_set: set          = set() # fast membership test

        # Fonts (created once, rebuilt by set_text_scale on scale change)
        self._font_name  = QFont("Segoe UI", 10)
        self._font_badge = QFont("Segoe UI", 7)
        self._font_group = QFont("Segoe UI", 11, QFont.Bold)
        self._font_scroll_hint = QFont("Segoe UI", 8)

        # Text-scale-aware geometry — these start as the module constants
        # but become per-instance so set_text_scale() can grow them
        # alongside the fonts (otherwise larger text at 200% would clip
        # against a label/header band sized for 100% text).
        self._label_h        = LABEL_H
        self._group_header_h = GROUP_HEADER_H
        self._rating_row_h   = RATING_ROW_H
        self._badge_r        = BADGE_R

        # Zoom debounce — Ctrl+scroll fires many events per gesture; the
        # expensive part of a zoom (cache clear, config save, refetch) only
        # runs once, shortly after scrolling stops. The cell size itself
        # updates immediately so the resize still feels responsive.
        self._zoom_commit_timer = QTimer(self)
        self._zoom_commit_timer.setSingleShot(True)
        self._zoom_commit_timer.setInterval(150)
        self._zoom_commit_timer.timeout.connect(self._commit_zoom)

        # Grouping / sorting — never persisted between sessions (by design)
        self._group_by: str      = 'none'   # 'none'|'date'|'type'|'size'|'favorites'
        self._sort_by:  str      = 'none'   # 'none'|'date'|'type'|'size'|'favorites'
        self._sort_ascending: bool = False  # default matches prior behavior (descending)
        self._collapsed_groups: set = set()  # group keys currently collapsed
        self._layout: List[dict] = []        # precomputed layout (see _rebuild_layout)
        self._layout_height: int = 0
        self._orig_index_cache = None
        self._orig_index_cache_len = -1

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)

    # ── Cell geometry ─────────────────────────────────────────────────────────
    #
    # Layout model: self._layout is a flat list of dicts, built once by
    # _rebuild_layout() whenever rows/grouping/sorting/zoom/width changes.
    # Each entry is either:
    #   {'kind':'header', 'key':str, 'label':str, 'y':int, 'h':int}
    #   {'kind':'item',   'row_idx':int, 'rect':QRect}   # rect in scroll-space
    # This lets grouping insert header bands without special-casing every
    # geometry function — they all just walk this list.
    #

    def _calc_cell(self) -> int:
        return max(int(THUMB_BASE*0.25),
                   min(int(THUMB_BASE*2.0), int(THUMB_BASE*self._scale)))

    def set_text_scale(self, scale: float):
        """
        Rebuilds every font and every text-dependent geometry value used
        by the painter-based grid (filename labels, badges, group headers,
        rating rows) at the new scale, then rebuilds the layout so nothing
        overlaps or clips. This is the GalleryCanvas half of the app-wide
        text-size setting — the QSS half (build_app_style) covers every
        other dialog automatically since they're real Qt widgets, but this
        canvas paints its own text directly and needs its own hook.
        """
        self._font_name  = QFont("Segoe UI", max(1, round(10 * scale)))
        self._font_badge = QFont("Segoe UI", max(1, round(7 * scale)))
        self._font_group = QFont("Segoe UI", max(1, round(11 * scale)), QFont.Bold)
        self._font_scroll_hint = QFont("Segoe UI", max(1, round(8 * scale)))
        self._label_h        = max(LABEL_H,        round(LABEL_H * scale))
        self._group_header_h = max(GROUP_HEADER_H, round(GROUP_HEADER_H * scale))
        self._rating_row_h   = max(RATING_ROW_H,   round(RATING_ROW_H * scale))
        self._badge_r        = max(BADGE_R,         round(BADGE_R * scale))
        self._rebuild_layout()
        self._update_scrollbar()
        self.viewport().update()

    def _cols(self) -> int:
        vw = self.viewport().width() or 800
        return max(1, (vw - CELL_PAD) // (self._cell_size + CELL_PAD))

    def _cell_w(self) -> int: return self._cell_size + CELL_PAD
    def _cell_h(self) -> int: return self._cell_size + self._label_h + CELL_PAD

    # Size buckets for grouping — granularity chosen to fit a typical mixed
    # photo/video/zip library: fine-grained at the small end (where most
    # photos/individual images land) and coarser at the large end (where
    # videos and zip archives tend to sit).
    SIZE_BUCKETS = [
        (5,     "0–5 MB"),
        (10,    "6–10 MB"),
        (20,    "11–20 MB"),
        (50,    "21–50 MB"),
        (100,   "51–100 MB"),
        (250,   "101–250 MB"),
        (500,   "251–500 MB"),
        (1000,  "501 MB–1 GB"),
        (None,  "1 GB+"),
    ]

    def _size_bucket(self, file_size) -> Tuple[int, str]:
        """Returns (bucket_index, label) for a file's size in bytes,
        per SIZE_BUCKETS. bucket_index is used as the sort key so buckets
        stay in size order regardless of label text."""
        mb = (file_size or 0) / (1024 * 1024)
        for i, (ceiling, label) in enumerate(self.SIZE_BUCKETS):
            if ceiling is None or mb <= ceiling:
                return i, label
        return len(self.SIZE_BUCKETS) - 1, self.SIZE_BUCKETS[-1][1]

    def _group_key_label(self, row) -> Tuple[str, str]:
        """Returns (sort_key, display_label) for the row's group, per _group_by."""
        if self._group_by == 'date':
            # date_modified (the file's own mtime) rather than date_added —
            # date_added gets reset to "now" whenever a file is freshly
            # imported or relinked by a rescan, making it meaningless as a
            # grouping signal over time. date_modified reflects the actual
            # file content's last-changed time and stays stable.
            d = (row['date_modified'] or '')[:10]  # YYYY-MM-DD
            return d, (d or 'Unknown date')
        if self._group_by == 'type':
            if row['file_type'] == 'folder':
                return '\x00folder', 'Folders'   # \x00 sorts first, ahead of any extension
            ext = Path(row['filename']).suffix.lower().lstrip('.')
            return ext, (ext.upper() if ext else 'No extension')
        if self._group_by == 'size':
            idx, label = self._size_bucket(row['file_size'])
            return f"{idx:02d}", label   # zero-padded index keeps buckets in size order
        if self._group_by == 'favorites':
            s = row['stars'] or 0; h = row['hearts'] or 0
            # One combined key per spec example: "Hearts:5", "Stars:4", etc.
            # Group primarily by hearts descending, ties broken by stars.
            key = f"{5-h:1d}{5-s:1d}"   # zero-padded for correct string sort (desc)
            label_parts = []
            if h: label_parts.append(f"Hearts:{h}")
            if s: label_parts.append(f"Stars:{s}")
            label = ", ".join(label_parts) if label_parts else "Unrated"
            return key, label
        return '', ''

    def _sort_rows(self, rows: list) -> list:
        """
        Apply current sort field and direction. Each field has a natural
        default direction (date/size/favorites: newest/largest/highest
        first, i.e. descending; type: A-Z, i.e. ascending) — the
        ascending/descending toggle flips relative to that natural
        default rather than forcing every field into the same absolute
        direction, so switching sort fields doesn't silently reverse a
        field whose sensible default was already ascending.
        """
        asc = self._sort_ascending
        if self._sort_by == 'date':
            # date_modified — see _group_key_label for why not date_added.
            # Natural default: newest first (descending).
            return sorted(rows, key=lambda r: r['date_modified'] or '', reverse=not asc)
        if self._sort_by == 'type':
            # Natural default: A-Z (ascending).
            return sorted(rows, key=lambda r: Path(r['filename']).suffix.lower(), reverse=asc)
        if self._sort_by == 'size':
            # Natural default: largest first (descending).
            return sorted(rows, key=lambda r: r['file_size'] or 0, reverse=not asc)
        if self._sort_by == 'favorites':
            # Natural default: highest-rated first (descending).
            return sorted(rows,
                key=lambda r: ((r['hearts'] or 0), (r['stars'] or 0)), reverse=not asc)
        return rows   # 'none' — keep DB order (type DESC, filename)

    def _rebuild_layout(self):
        """
        Recompute self._layout from self._rows, current group_by/sort_by/
        collapsed state, and current column count. Must be called whenever
        any of those change, or the viewport is resized (column count).
        """
        cols = self._cols()
        cw, ch = self._cell_w(), self._cell_h()
        rows = self._sort_rows(self._rows)

        layout = []
        y = CELL_PAD

        if self._group_by == 'none':
            for i, row in enumerate(rows):
                col_n = i % cols
                row_n = i // cols
                if col_n == 0 and row_n > 0:
                    pass  # y advances naturally below via row_n calc
                rx = CELL_PAD + col_n * cw
                ry = CELL_PAD + row_n * ch
                layout.append({'kind':'item', 'row_idx': self._orig_index(row),
                               'rect': QRect(rx, ry, self._cell_size, self._cell_size+self._label_h)})
            y = CELL_PAD + ((len(rows)+cols-1)//cols) * ch if rows else 0

        else:
            # Group rows while preserving sort order within each group
            groups: Dict[str, dict] = {}
            order: List[str] = []
            for row in rows:
                key, label = self._group_key_label(row)
                if key not in groups:
                    groups[key] = {'label': label, 'rows': []}
                    order.append(key)
                groups[key]['rows'].append(row)

            for key in order:
                g = groups[key]
                layout.append({'kind':'header', 'key':key,
                               'label': f"{g['label']}  ({len(g['rows'])})",
                               'y': y, 'h': self._group_header_h})
                y += self._group_header_h
                if key in self._collapsed_groups:
                    continue   # skip items — group is collapsed
                for i, row in enumerate(g['rows']):
                    col_n = i % cols
                    row_n = i // cols
                    rx = CELL_PAD + col_n * cw
                    ry = y + row_n * ch
                    layout.append({'kind':'item', 'row_idx': self._orig_index(row),
                                   'rect': QRect(rx, ry, self._cell_size, self._cell_size+self._label_h)})
                n_rows = (len(g['rows'])+cols-1)//cols
                y += n_rows * ch

        self._layout = layout
        self._layout_height = y + CELL_PAD

    def _orig_index(self, row) -> int:
        """Map a (possibly sorted/grouped) row back to its index in self._rows.
        Cached via id->index dict, rebuilt lazily when stale."""
        if (self._orig_index_cache is None
                or self._orig_index_cache_len != len(self._rows)):
            self._orig_index_cache = {r['id']: i for i, r in enumerate(self._rows)}
            self._orig_index_cache_len = len(self._rows)
        return self._orig_index_cache.get(row['id'], -1)

    def _total_height(self) -> int:
        return getattr(self, '_layout_height', 0)

    def _idx_at(self, x: int, y: int) -> int:
        """Row-index (into self._rows) of the item at viewport coords (x,y).
        Returns -1 if the click hit empty space or a header band."""
        vy = y + self.verticalScrollBar().value()
        for entry in self._layout:
            if entry['kind'] != 'item': continue
            if entry['rect'].contains(x, vy):
                return entry['row_idx']
        return -1

    def _header_at(self, x: int, y: int) -> Optional[str]:
        """Group key of the header band at viewport coords (x,y), else None."""
        vy = y + self.verticalScrollBar().value()
        for entry in self._layout:
            if entry['kind'] != 'header': continue
            if entry['y'] <= vy <= entry['y'] + entry['h']:
                return entry['key']
        return None

    def _cell_rect_for_row_idx(self, row_idx: int) -> Optional[QRect]:
        """Rectangle (scroll-space) for the item currently at self._rows[row_idx]."""
        for entry in self._layout:
            if entry['kind'] == 'item' and entry['row_idx'] == row_idx:
                return entry['rect']
        return None

    def _ratings_clickable(self) -> bool:
        """Below RATING_MIN_CELL px, stars/hearts are view-only (too small to click)."""
        return self._cell_size >= RATING_MIN_CELL

    def _rating_hit_test(self, row_idx: int, vx: int, vy: int) -> Optional[Tuple[str,int]]:
        """
        vx,vy are VIEWPORT coordinates (already scroll-adjusted, same space
        the mouse handlers work in). Returns ('star', 1-5) or ('heart', 1-5)
        if the click landed on a rating icon, else None.
        Only active when _ratings_clickable() — small cells are view-only.
        """
        if not self._ratings_clickable():
            return None
        cell = self._cell_rect_for_row_idx(row_idx)
        if cell is None:
            return None
        sv   = self.verticalScrollBar().value()
        cell_vx = cell.x()
        cell_vy = cell.y() - sv
        thumb_x = cell_vx + CELL_PAD//2
        thumb_y = cell_vy + CELL_PAD//2

        if not (thumb_y <= vy <= thumb_y + self._rating_row_h):
            return None

        icon_w = RATING_ICON_SZ
        # Stars: top-left, 5 slots
        star_start = thumb_x + 2
        if star_start <= vx <= star_start + icon_w*5:
            n = int((vx - star_start) // icon_w) + 1
            return ('star', max(1, min(5, n)))
        # Hearts: top-right, 5 slots
        heart_end   = thumb_x + self._cell_size - 2
        heart_start = heart_end - icon_w*5
        if heart_start <= vx <= heart_end:
            n = int((vx - heart_start) // icon_w) + 1
            return ('heart', max(1, min(5, n)))
        return None

    def _visible_range(self) -> Tuple[int,int]:
        """First and last+1 ROW indices (into self._rows) currently visible
        on screen, based on the precomputed layout. Used to decide which
        thumbnails to fetch — order doesn't matter for that purpose."""
        sv = self.verticalScrollBar().value()
        vh = self.viewport().height()
        clip_top, clip_bottom = sv, sv + vh
        visible_idxs = []
        for entry in self._layout:
            if entry['kind'] != 'item': continue
            r = entry['rect']
            if r.y() + r.height() < clip_top: continue
            if r.y() > clip_bottom: break
            visible_idxs.append(entry['row_idx'])
        if not visible_idxs:
            return 0, 0
        return min(visible_idxs), max(visible_idxs)+1

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def current_parent_id(self) -> Optional[int]:
        return self._parent_stack[-1][0]

    def breadcrumb_path(self) -> List[Tuple[Optional[int],str]]:
        return list(self._parent_stack)

    def set_scale(self, scale: float):
        """Full zoom commit: used by the toolbar slider (already low-frequency).
        Clears caches, persists to config, and re-fetches immediately."""
        self._scale     = scale
        self._cell_size = self._calc_cell()
        self._pixmap_cache.clear()     # invalidate — wrong size
        self._frame_cache.clear()
        self._rebuild_layout()
        self._update_scrollbar()
        self.viewport().update()
        self.config.set('thumb_scale', str(scale))
        # Re-fetch visible items at new size
        self._fetch_visible()

    def _preview_scale(self, scale: float):
        """Cheap, instant resize used during Ctrl+scroll: updates cell size
        and repaints so the zoom feels live, but skips cache clearing, disk
        writes, and re-fetching — those happen once in _commit_zoom after
        scrolling settles. Existing pixmaps just get scaled to the new size
        by the paint routine's drawPixmap centering (slightly soft on the
        way to the target size, sharp again once _commit_zoom runs)."""
        self._scale     = scale
        self._cell_size = self._calc_cell()
        self._rebuild_layout()
        self._update_scrollbar()
        self.viewport().update()

    def _commit_zoom(self):
        """Runs once, ~150ms after the last Ctrl+scroll notch. Does the
        expensive part: cache invalidation, config save, re-fetch at the
        now-final cell size."""
        self._pixmap_cache.clear()
        self._frame_cache.clear()
        self._rebuild_layout()
        self.viewport().update()
        self.config.set('thumb_scale', str(self._scale))
        self._fetch_visible()

    def load_view(self, rows: List):
        """Replace current view. No widget creation — just stores rows and repaints."""
        self._cancel_fetchers()
        self._rows         = list(rows)
        self._orig_index_cache = None   # invalidate — row identities changed
        self._collapsed_groups.clear()  # grouping never persists across a reload
        self._selected.clear()
        self._last_click_idx = None
        self._hover_idx    = -1
        self._hover_frame  = 0
        self._hover_timer.stop()
        self._pixmap_cache.clear()
        self._frame_cache.clear()
        self._frame_counts.clear()
        self._hover_loading.clear()
        self._fetch_queue.clear()
        self._fetch_queue_set.clear()
        for loader in self._hover_loaders.values():
            if loader.isRunning():
                loader.quit(); loader.wait(200)
        self._hover_loaders.clear()
        self.selection_changed.emit([])
        self._rebuild_layout()
        self._update_scrollbar()
        self.verticalScrollBar().setValue(0)
        self.viewport().update()
        self._fetch_visible()
        log.debug("load_view: %d rows", len(rows))

    def get_selected_ids(self) -> List[int]:
        return list(self._selected)

    def refresh_rows_from_db(self, ids: List[int]):
        """Re-read specific rows from DB (e.g. after a rating change) and repaint,
        without a full load_view() — keeps scroll position and pixmap cache.
        If grouping/sorting is active, the row's group/position may change
        (e.g. re-rating moves it in a favorites sort), so layout is rebuilt."""
        if not self.db: return
        id_set = set(ids)
        for i, row in enumerate(self._rows):
            if row['id'] in id_set:
                fresh = self.db.get_file_by_id(row['id'])
                if fresh:
                    self._rows[i] = fresh
        if self._group_by != 'none' or self._sort_by != 'none':
            self._rebuild_layout()
            self._update_scrollbar()
        self.viewport().update()

    def jump_to_stack_index(self, idx: int):
        self._parent_stack = self._parent_stack[:idx+1]
        self._reload_current()

    def _reload_current(self):
        if not self.db: return
        rows = self.db.get_files(parent_id=self.current_parent_id)
        self.load_view(rows)
        self.navigate_folder.emit(
            self.current_parent_id if self.current_parent_id is not None else -1,
            self._parent_stack[-1][1])

    # ── Scrollbar ─────────────────────────────────────────────────────────────

    def _update_scrollbar(self):
        h = self._total_height()
        vh = self.viewport().height()
        sb = self.verticalScrollBar()
        sb.setRange(0, max(0, h - vh))
        sb.setPageStep(vh)
        sb.setSingleStep(self._cell_h())

    def resizeEvent(self, event):
        self._rebuild_layout()   # column count depends on viewport width
        self._update_scrollbar()
        self.viewport().update()
        # The very first load (MainWindow._init_library, before win.show())
        # can run while the widget still reports a default/unlaid-out size,
        # so _fetch_visible() at that point may only "see" a handful of
        # rows as visible and never revisit the rest until the user
        # scrolls. Qt fires a real resizeEvent once the window is actually
        # shown at its true size, so re-fetching here catches that gap —
        # and also keeps things correct on any later manual resize.
        self._fetch_visible()
        super().resizeEvent(event)

    def _on_scroll(self, _val):
        self.viewport().update()
        self._fetch_visible()

    # ── Fetcher management ────────────────────────────────────────────────────

    def _cancel_hover_loaders(self):
        for loader in list(self._hover_loaders.values()):
            if loader.isRunning():
                loader.quit(); loader.wait(200)
        self._hover_loaders.clear()
        self._hover_loading.clear()

    def _cancel_fetchers(self):
        for f in self._fetchers:
            if f.isRunning():
                f.cancel(); f.wait(200)
        self._fetchers.clear()
        self._fetch_queue.clear()
        self._fetch_queue_set.clear()

    def _fetch_visible(self):
        if not self.db: return
        first, last = self._visible_range()
        # Preload a fixed number of extra rows beyond the visible area.
        # PRELOAD_ROWS is independent of library size so behaviour is
        # consistent whether the library has 300 or 100k items.
        extra = self._cols() * PRELOAD_ROWS
        end   = min(len(self._rows), last + extra)
        ids   = [self._rows[i]['id'] for i in range(first, end)
                 if self._rows[i]['id'] not in self._pixmap_cache]
        if not ids: return
        self._start_fetcher(ids)

    def _start_fetcher(self, ids: List[int]):
        """Start at most ONE ThumbFetcher at a time.
        Pending IDs are queued; next fetch starts automatically
        when the current one finishes via _on_fetcher_done."""
        if not ids or not self.db: return
        # Deduplicate: skip ids already cached or already queued
        new_ids = [i for i in ids
                   if i not in self._pixmap_cache
                   and i not in self._fetch_queue_set]
        if not new_ids: return
        self._fetch_queue.extend(new_ids)
        self._fetch_queue_set.update(new_ids)
        # Only launch if nothing is currently running
        if not any(f.isRunning() for f in self._fetchers):
            self._launch_next_fetcher()

    def _launch_next_fetcher(self):
        if not self._fetch_queue or not self.db: return
        batch = self._fetch_queue[:200]          # max 200 per fetch run
        self._fetch_queue = self._fetch_queue[200:]
        f = ThumbFetcher(self.db.lib_path, batch)
        f.result.connect(self._on_thumb_result, Qt.QueuedConnection)
        f.finished.connect(self._on_fetcher_done, Qt.QueuedConnection)
        f.start()
        self._fetchers = [f]                     # only ever 1 active fetcher
        log.debug("ThumbFetcher started: %d ids, %d queued", len(batch), len(self._fetch_queue))

    def _on_fetcher_done(self):
        """Called when the active ThumbFetcher finishes. Start next batch if queued."""
        self._fetchers = [f for f in self._fetchers if f.isRunning()]
        if self._fetch_queue:
            self._launch_next_fetcher()

    @pyqtSlot(int, object, int)
    def _on_thumb_result(self, fid: int, qimg, frame_count: int):
        """Main thread only. Convert QImage → QPixmap, store in cache, repaint."""
        self._fetch_queue_set.discard(fid)   # no longer pending
        self._frame_counts[fid] = frame_count
        if qimg is not None:
            # Scale to cell size maintaining aspect ratio
            scaled = qimg.scaled(self._cell_size, self._cell_size,
                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._pixmap_cache[fid] = QPixmap.fromImage(scaled)
        else:
            # Placeholder
            ftype = 'image'
            for r in self._rows:
                if r['id'] == fid: ftype = r['file_type']; break
            pimg = placeholder_qimage(self._cell_size, self._cell_size,
                                      BADGE_ICONS.get(ftype,'?'))
            self._pixmap_cache[fid] = QPixmap.fromImage(pimg)
        self.viewport().update()

    # ── Hover animation ───────────────────────────────────────────────────────

    def _advance_hover(self):
        if self._hover_idx < 0 or self._hover_idx >= len(self._rows):
            self._hover_timer.stop(); return
        fid    = self._rows[self._hover_idx]['id']
        frames = self._frame_cache.get(fid)
        if not frames:
            return   # still loading — timer keeps running, we just skip this tick
        self._hover_frame = (self._hover_frame + 1) % len(frames)
        self.viewport().update()

    def _load_hover_frames(self, fid: int):
        """
        Start a HoverLoader QThread for this fid.
        Guarded by _hover_loading set to prevent duplicate threads.
        frames_ready signal is connected with Qt.QueuedConnection so
        _on_hover_frames always runs on the main thread — the only safe
        place to create QPixmap objects.
        """
        if fid in self._hover_loading:
            return
        self._hover_loading.add(fid)
        loader = HoverLoader(self.db.lib_path, fid, self._cell_size)
        loader.frames_ready.connect(self._on_hover_frames, Qt.QueuedConnection)
        loader.start()
        # Keep reference so it isn't garbage-collected mid-run
        self._hover_loaders[fid] = loader

    @pyqtSlot(int, list)
    def _on_hover_frames(self, fid: int, qimages: list):
        """Main thread only — guaranteed by QueuedConnection."""
        self._hover_loading.discard(fid)
        self._hover_loaders.pop(fid, None)
        # Convert QImage → QPixmap here (main thread only)
        pixmaps = [QPixmap.fromImage(qi) for qi in qimages]
        self._frame_cache[fid] = pixmaps
        # Start animation if we're still hovering this item
        if (self._hover_idx >= 0
                and self._hover_idx < len(self._rows)
                and self._rows[self._hover_idx]['id'] == fid):
            self._hover_frame = 0
            self._hover_timer.start()
            self.viewport().update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        if not self._rows:
            self._paint_empty(); return

        p      = QPainter(self.viewport())
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        sv  = self.verticalScrollBar().value()
        cs  = self._cell_size
        vw  = self.viewport().width()
        vh  = self.viewport().height()

        clip_top    = sv
        clip_bottom = sv + vh

        p.fillRect(0, 0, vw, vh, QColor(DARK_BG))

        fm = QFontMetrics(self._font_name)

        for entry in self._layout:
            if entry['kind'] == 'header':
                hy = entry['y']
                if hy + entry['h'] < clip_top:  continue
                if hy > clip_bottom:            break
                self._paint_group_header(p, entry, hy - sv, vw)
                continue

            # entry['kind'] == 'item'
            r = entry['rect']
            if r.y() + r.height() < clip_top:  continue
            if r.y() > clip_bottom:            break

            row = self._rows[entry['row_idx']]
            vx  = r.x()
            vy  = r.y() - sv

            fid    = row['id']
            ftype  = row['file_type']
            fname  = row['filename']
            sel    = fid in self._selected
            hover  = (entry['row_idx'] == self._hover_idx)

            # ── Card background ──────────────────────────────────────────────
            card_rect = QRect(vx, vy, cs + CELL_PAD//2, cs + self._label_h)
            bg = QColor(CARD_HOV if hover else CARD_BG)
            border_col = QColor(SEL_COL if sel else (ACCENT2 if hover else BORDER))
            border_w   = 2 if sel else 1
            p.setBrush(QBrush(bg))
            p.setPen(QPen(border_col, border_w))
            p.drawRoundedRect(card_rect, 7, 7)

            # ── Thumbnail ────────────────────────────────────────────────────
            pix = None
            if hover and ftype == 'video':
                frames = self._frame_cache.get(fid)
                if frames and self._hover_frame < len(frames):
                    pix = frames[self._hover_frame]
            if pix is None:
                pix = self._pixmap_cache.get(fid)

            thumb_rect = QRect(vx + CELL_PAD//2, vy + CELL_PAD//2, cs, cs)
            if pix:
                pw = pix.width(); ph = pix.height()
                ox = (cs - pw) // 2
                oy = (cs - ph) // 2
                p.drawPixmap(vx + CELL_PAD//2 + ox, vy + CELL_PAD//2 + oy, pix)
            else:
                p.fillRect(thumb_rect, QColor(CARD_BG))
                p.setPen(QColor(TEXT_SEC))
                if fid in self._fetch_queue_set:
                    # Actively queued — a fetch is already in flight for
                    # this card, it just hasn't come back yet.
                    p.setFont(self._font_badge)
                    p.drawText(thumb_rect, Qt.AlignCenter, "…")
                else:
                    # Not requested at all yet (below the initial preload
                    # range) — say so explicitly rather than leaving an
                    # ambiguous blank/spinner that never resolves until
                    # the user happens to scroll near it.
                    p.setFont(self._font_scroll_hint)
                    p.drawText(thumb_rect, Qt.AlignCenter | Qt.TextWordWrap,
                              "Scroll to\nload")

            # ── Type badge ───────────────────────────────────────────────────
            bx = vx + CELL_PAD//2 + cs - self._badge_r - 3
            by = vy + CELL_PAD//2 + cs - self._badge_r - 3
            p.setBrush(QBrush(QColor(BADGE_COLS.get(ftype, TEXT_SEC))))
            p.setPen(Qt.NoPen)
            p.drawEllipse(bx, by, self._badge_r, self._badge_r)
            p.setPen(QColor('white'))
            p.setFont(self._font_badge)
            p.drawText(QRect(bx, by, self._badge_r, self._badge_r),
                       Qt.AlignCenter, BADGE_ICONS.get(ftype,'?'))

            # ── Item count badge (zip/folder only) ──────────────────────────────
            # Bottom-left corner, opposite the type badge — shows how many
            # top-level items are inside without needing to open it.
            item_count = row['item_count'] if 'item_count' in row.keys() else None
            if ftype in ('zip', 'folder') and item_count:
                count_text = str(item_count) if item_count < 1000 else "999+"
                fm_badge = QFontMetrics(self._font_badge)
                text_w = fm_badge.horizontalAdvance(count_text)
                pad = 5
                pill_w = text_w + pad*2
                pill_h = self._badge_r
                px = vx + CELL_PAD//2 + 3
                py = vy + CELL_PAD//2 + cs - pill_h - 3
                p.setBrush(QBrush(QColor(0, 0, 0, 170)))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(QRect(px, py, pill_w, pill_h), pill_h//2, pill_h//2)
                p.setPen(QColor('white'))
                p.setFont(self._font_badge)
                p.drawText(QRect(px, py, pill_w, pill_h), Qt.AlignCenter, count_text)

            # ── Ratings row (stars top-left, hearts top-right) ─────────────────
            stars  = row['stars']  or 0
            hearts = row['hearts'] or 0
            thumb_x = vx + CELL_PAD//2
            thumb_y = vy + CELL_PAD//2
            if stars or hearts or hover:
                if self._ratings_clickable():
                    self._paint_rating_icons(p, thumb_x, thumb_y, stars, hearts)
                else:
                    self._paint_rating_text(p, thumb_x, thumb_y, cs, stars, hearts)

            # ── Filename label ───────────────────────────────────────────────
            lbl_rect = QRect(vx, vy + cs + CELL_PAD//2 + 2, cs + CELL_PAD//2, self._label_h - 6)
            p.setPen(QColor(TEXT_SEC))
            p.setFont(self._font_name)
            elided = fm.elidedText(fname, Qt.ElideMiddle, cs - 4)
            p.drawText(lbl_rect, Qt.AlignHCenter | Qt.AlignTop, elided)

        p.end()

    def _paint_group_header(self, p: QPainter, entry: dict, vy: int, vw: int):
        """Draw a collapsible group header band: chevron + label + item count."""
        key       = entry['key']
        collapsed = key in self._collapsed_groups
        band_rect = QRect(0, vy, vw, entry['h'])

        p.fillRect(band_rect, QColor(PANEL_BG))
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawLine(0, vy + entry['h'] - 1, vw, vy + entry['h'] - 1)

        # Chevron
        chevron = "▸" if collapsed else "▾"
        p.setPen(QColor(TEXT_PRI))
        p.setFont(self._font_group)
        p.drawText(QRect(CELL_PAD, vy, 24, entry['h']),
                   Qt.AlignVCenter | Qt.AlignLeft, chevron)

        # Label
        p.drawText(QRect(CELL_PAD + 24, vy, vw - CELL_PAD*2 - 24, entry['h']),
                   Qt.AlignVCenter | Qt.AlignLeft, entry['label'])

    def _paint_rating_icons(self, p: QPainter, thumb_x: int, thumb_y: int,
                             stars: int, hearts: int):
        """Draw 5 clickable star slots (top-left) and 5 heart slots (top-right)."""
        icon_w = RATING_ICON_SZ
        f = QFont("Segoe UI", max(int(icon_w*0.62), 7))
        p.setFont(f)

        # Semi-transparent backing strip so icons read on any thumbnail
        p.fillRect(QRect(thumb_x, thumb_y, self._cell_size, self._rating_row_h),
                   QColor(0, 0, 0, 110))

        # Stars — top-left
        sx = thumb_x + 2
        for i in range(5):
            filled = i < stars
            p.setPen(QColor("#ffd54a") if filled else QColor(255,255,255,90))
            glyph = "★" if filled else "☆"
            p.drawText(QRect(sx + i*icon_w, thumb_y, icon_w, self._rating_row_h),
                       Qt.AlignCenter, glyph)

        # Hearts — top-right
        hx = thumb_x + self._cell_size - 2 - icon_w*5
        for i in range(5):
            filled = i < hearts
            p.setPen(QColor("#ff5f7a") if filled else QColor(255,255,255,90))
            glyph = "♥" if filled else "♡"
            p.drawText(QRect(hx + i*icon_w, thumb_y, icon_w, self._rating_row_h),
                       Qt.AlignCenter, glyph)

    def _paint_rating_text(self, p: QPainter, thumb_x: int, thumb_y: int,
                            cs: int, stars: int, hearts: int):
        """View-only compact form for small cells: e.g. '5♥ 3★'."""
        if not stars and not hearts:
            return
        text = ""
        if hearts: text += f"{hearts}♥ "
        if stars:  text += f"{stars}★"
        text = text.strip()
        if not text: return
        p.fillRect(QRect(thumb_x, thumb_y, cs, self._rating_row_h), QColor(0,0,0,110))
        p.setPen(QColor("white"))
        p.setFont(self._font_badge)
        p.drawText(QRect(thumb_x, thumb_y, cs, self._rating_row_h), Qt.AlignCenter, text)

    def _paint_empty(self):
        p = QPainter(self.viewport())
        p.fillRect(self.viewport().rect(), QColor(DARK_BG))
        p.setPen(QColor(TEXT_SEC))
        p.setFont(QFont("Segoe UI", fs(14)))
        p.drawText(self.viewport().rect(), Qt.AlignCenter,
                   "No items to display.\nUse ➕ Add Folder.")
        p.end()

    # ── Event filter (viewport mouse events) ──────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.viewport():
            t = event.type()
            if t == event.MouseButtonPress:
                self._mouse_press(event)
            elif t == event.MouseButtonDblClick:
                self._mouse_dbl(event)
            elif t == event.MouseMove:
                self._mouse_move(event)
            elif t == event.Leave:
                self._mouse_leave()
            elif t == event.Wheel:
                if event.modifiers() & Qt.ControlModifier:
                    self._wheel_zoom(event)
                    return True   # consume — don't also scroll the list
        return super().eventFilter(obj, event)

    def _wheel_zoom(self, event):
        """Ctrl+scroll: zoom in/out like Windows Explorer, 1% per notch for
        fine control. Uses the cheap preview path so rapid scrolling stays
        smooth; the expensive cache/refetch work is debounced via
        _zoom_commit_timer and only runs once scrolling settles."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        cur_pct  = int(self._scale * 100)
        step     = 1
        new_pct  = cur_pct + (step if delta > 0 else -step)
        new_pct  = max(25, min(200, new_pct))
        if new_pct == cur_pct:
            return
        self._preview_scale(new_pct / 100.0)
        self.zoom_changed.emit(new_pct)
        self._zoom_commit_timer.start()   # (re)start — coalesces rapid notches

    def _mouse_press(self, event):
        # ── Group header click (collapse/expand) ────────────────────────────────
        if event.button() == Qt.LeftButton:
            header_key = self._header_at(event.x(), event.y())
            if header_key is not None:
                if header_key in self._collapsed_groups:
                    self._collapsed_groups.discard(header_key)
                else:
                    self._collapsed_groups.add(header_key)
                self._rebuild_layout()
                self._update_scrollbar()
                self.viewport().update()
                return

        idx = self._idx_at(event.x(), event.y())
        if idx < 0:
            self._clear_sel(); return

        fid  = self._rows[idx]['id']
        mods = event.modifiers()

        # ── Rating click (stars/hearts) takes priority over selection ──────────
        if event.button() == Qt.LeftButton:
            hit = self._rating_hit_test(idx, event.x(), event.y())
            if hit:
                kind, n = hit
                cur_row = self._rows[idx]
                current = (cur_row['stars'] if kind=='star' else cur_row['hearts']) or 0
                # Clicking the currently-set top value again does NOT clear it —
                # per spec, zeroing out requires the right-click "Reset Ratings"
                # action. Clicking any star/heart just sets the rating to that count.
                new_val = n
                if kind == 'star':
                    self.db.set_stars(fid, new_val)
                else:
                    self.db.set_hearts(fid, new_val)
                self._rows[idx] = self.db.get_file_by_id(fid)
                # A rating change can move this item to a different group or
                # sort position when Favorites grouping/sorting is active
                # (same situation refresh_rows_from_db already handles for
                # the Reset Ratings action) — rebuild the layout so the
                # card doesn't sit in a now-stale position.
                if self._group_by != 'none' or self._sort_by != 'none':
                    self._rebuild_layout()
                    self._update_scrollbar()
                self.viewport().update()
                return

        if event.button() == Qt.RightButton:
            if fid not in self._selected:
                self._clear_sel(); self._selected=[fid]
            self.viewport().update()
            self.selection_changed.emit(list(self._selected))
            self.item_right_click.emit(fid)
            return

        if mods & Qt.ControlModifier:
            if fid in self._selected: self._selected.remove(fid)
            else: self._selected.append(fid)
            self._last_click_idx = idx
        elif mods & Qt.ShiftModifier and self._last_click_idx is not None:
            lo=min(self._last_click_idx,idx); hi=max(self._last_click_idx,idx)
            for i in range(lo, hi+1):
                f=self._rows[i]['id']
                if f not in self._selected: self._selected.append(f)
        else:
            self._clear_sel()
            self._selected=[fid]
            self._last_click_idx=idx

        self.viewport().update()
        self.selection_changed.emit(list(self._selected))

    def _mouse_dbl(self, event):
        if event.button() != Qt.LeftButton: return
        if self._header_at(event.x(), event.y()) is not None:
            return   # double-click on a header does nothing extra
        idx = self._idx_at(event.x(), event.y())
        if idx < 0: return
        row = self._rows[idx]
        if row['file_type'] == 'folder':
            self._parent_stack.append((row['id'], row['filename']))
            self._reload_current()
        else:
            self.item_double_click.emit(row['id'])

    def _mouse_move(self, event):
        idx = self._idx_at(event.x(), event.y())

        # ── Filename tooltip — checked on every move, even within the same
        # item, since the label sits below the thumbnail (different region).
        self._update_filename_tooltip(idx, event)

        if idx == self._hover_idx: return
        self._hover_idx   = idx
        self._hover_frame = 0
        self._hover_timer.stop()

        if idx >= 0:
            fid   = self._rows[idx]['id']
            ftype = self._rows[idx]['file_type']
            if ftype == 'video':
                if fid in self._frame_cache:
                    # Frames already cached — start slideshow immediately
                    self._hover_timer.start()
                else:
                    # Load frames; _on_hover_frames starts timer when ready
                    self._load_hover_frames(fid)

        self.viewport().update()

    def _update_filename_tooltip(self, idx: int, event):
        """Show the file's name as a tooltip when the cursor is over the
        filename label region of a card — useful at low zoom where the
        elided text on the card is very short. Capped at 128 characters
        (still elided further if the name itself is longer)."""
        if idx < 0:
            QToolTip.hideText()
            return
        cell = self._cell_rect_for_row_idx(idx)
        if cell is None:
            QToolTip.hideText()
            return
        sv = self.verticalScrollBar().value()
        label_top = (cell.y() - sv) + self._cell_size + CELL_PAD//2 + 2
        label_bottom = label_top + (self._label_h - 6)
        if not (label_top <= event.y() <= label_bottom):
            QToolTip.hideText()
            return
        fname = self._rows[idx]['filename']
        shown = fname if len(fname) <= 128 else fname[:125] + "..."
        QToolTip.showText(event.globalPos(), shown, self.viewport())

    def _mouse_leave(self):
        self._hover_idx = -1
        self._hover_frame = 0
        self._hover_timer.stop()
        QToolTip.hideText()
        self.viewport().update()

    def _clear_sel(self):
        self._selected.clear()
        self.selection_changed.emit([])


# ── Bulk tag assignment dialog ────────────────────────────────────────────────

class PickerThumb(QFrame):
    """
    One file card in BulkTagAssignDialog's picker grid — thumbnail, elided
    filename, and a checkbox overlay top-right. Clicking anywhere on the
    card toggles the checkbox (checkbox itself isn't a separate hit target,
    keeping the card easy to click at any zoom level).
    Size is instance-configurable via set_size() so the zoom slider can
    resize all cards live without leaking state across dialog instances.
    """
    toggled = pyqtSignal(int, bool)   # file_id, new_checked_state

    DEFAULT_SIZE = 130

    def __init__(self, file_id: int, filename: str, size: int = DEFAULT_SIZE, parent=None):
        super().__init__(parent)
        self.file_id  = file_id
        self.filename = filename
        self.checked  = False
        self.size_px  = size
        self._qimg: Optional[QImage] = None
        self.setCursor(Qt.PointingHandCursor)
        self._build()
        self.set_size(size)

    def _build(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(4,4,4,4); layout.setSpacing(3)
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet(f"background:{CARD_BG}; border-radius:4px;")
        self.thumb_label.setText("…")
        layout.addWidget(self.thumb_label)
        self.name_lbl = QLabel(self.filename)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px; background:transparent;")
        layout.addWidget(self.name_lbl)
        self._update_style()

    def set_size(self, size: int):
        """Live-resize this card (used by the zoom slider). Re-scales from
        the retained source QImage so quality doesn't degrade on zoom."""
        self.size_px = size
        self.setFixedSize(size, size + 24)
        self.thumb_label.setFixedSize(size-8, size-8)
        fm = QFontMetrics(self.name_lbl.font())
        self.name_lbl.setText(fm.elidedText(self.filename, Qt.ElideMiddle, size-8))
        if self._qimg is not None:
            self._render_thumb()

    def set_thumbnail(self, qimg: Optional[QImage]):
        self._qimg = qimg
        self._render_thumb()

    def _render_thumb(self):
        if self._qimg is not None:
            pix = QPixmap.fromImage(self._qimg).scaled(
                self.size_px-8, self.size_px-8, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumb_label.setPixmap(pix)
        else:
            self.thumb_label.setPixmap(QPixmap())
            self.thumb_label.setText("✕")

    def set_checked(self, checked: bool, emit: bool = False):
        self.checked = checked
        self._update_style()
        self.update()
        if emit:
            self.toggled.emit(self.file_id, checked)

    def _update_style(self):
        border = SEL_COL if self.checked else BORDER
        border_w = 2 if self.checked else 1
        self.setStyleSheet(f"""
            PickerThumb {{ background:{CARD_BG}; border:{border_w}px solid {border}; border-radius:6px; }}
            PickerThumb:hover {{ border-color:{ACCENT2}; }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.set_checked(not self.checked, emit=True)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = 18
        bx = self.width() - r - 4
        by = 4
        if self.checked:
            p.setBrush(QBrush(QColor(SUCCESS)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(bx, by, r, r)
            p.setPen(QPen(QColor('white'), 2))
            p.drawLine(bx+3, by+9, bx+7, by+13)
            p.drawLine(bx+7, by+13, bx+15, by+4)
        else:
            p.setBrush(QBrush(QColor(0,0,0,90)))
            p.setPen(QPen(QColor(TEXT_SEC), 1))
            p.drawEllipse(bx, by, r, r)
        p.end()


class TagPillBar(QWidget):
    """
    Horizontal row of removable 'pill' chips showing the currently-selected
    tags for a bulk operation. Each pill has an inline '×' to deselect it.
    Purely a display/removal widget — adding tags happens via the filter
    dropdown next to it (wired by the parent dialog).
    """
    tag_removed = pyqtSignal(tuple)   # (category, subcategory)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = FlowGridLayout(self, h_spacing=6, v_spacing=6, margin=4)
        self._pills: Dict[Tuple[str,str], QWidget] = {}

    def set_tags(self, tags: List[Tuple[str,str]]):
        self._layout.clear()
        self._pills.clear()
        for tag in tags:
            self._layout.addWidget(self._make_pill(tag))
        self.updateGeometry()

    def _make_pill(self, tag: Tuple[str,str]) -> QWidget:
        cat, sub = tag
        pill = QFrame()
        pill.setStyleSheet(f"""
            QFrame {{ background:{ACCENT}; border-radius:11px; }}
            QLabel {{ background:transparent; color:white; font-size:{fs(12)}px; }}
        """)
        pl = QHBoxLayout(pill); pl.setContentsMargins(10,4,6,4); pl.setSpacing(6)
        lbl = QLabel(f"{cat}: {sub}")
        pl.addWidget(lbl)
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(ws(18),ws(18))
        close_btn.setStyleSheet(
            f"QPushButton{{background:rgba(255,255,255,40);color:white;border:none;"
            f"border-radius:{ws(9)}px;font-size:{fs(10)}px;}}"
            f"QPushButton:hover{{background:rgba(255,255,255,80);}}")
        close_btn.clicked.connect(lambda: self.tag_removed.emit(tag))
        pl.addWidget(close_btn)
        self._pills[tag] = pill
        return pill


class BulkTagAssignDialog(QDialog):
    """
    Assign one or more tags to many files at once, additively — per spec,
    this never removes a file's existing tags, it only adds the selected
    ones on top (e.g. a zip already tagged 'car, sport' gains 'white' and
    'black' without losing the first two).

    - Tag picker: type to filter the vocabulary, pick multiple tags shown
      as removable pills.
    - File picker: thumbnail grid (reuses ThumbFetcher for background
      loading), library-wide, with a filename search box.
    - Apply Changes assigns every selected tag to every checked file.
    """

    MIN_ZOOM, MAX_ZOOM, DEFAULT_ZOOM = 90, 220, PickerThumb.DEFAULT_SIZE

    def __init__(self, db: LibraryDB, preset_tag: Optional[Tuple[str,str]] = None,
                 parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Bulk Assign Tags")
        self.setMinimumSize(820, 620)
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)

        self._rows: List = []
        self._cards: Dict[int, PickerThumb] = {}
        self._checked: set = set()
        self._selected_tags: List[Tuple[str,str]] = []
        self._all_vocab: List[Tuple[str,str]] = []
        self._fetcher: Optional[ThumbFetcher] = None

        self._build_ui()
        self._load_vocab()
        if preset_tag:
            self._add_tag_to_selection(preset_tag)
        self._refresh_files()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16,16,16,16); outer.setSpacing(10)

        title = QLabel("Bulk Assign Tags")
        title.setStyleSheet(f"color:{TEXT_PRI}; font-size:16px; font-weight:bold; background:transparent;")
        outer.addWidget(title)
        hint = QLabel("Adds the selected tags on top of whatever tags each file already has.")
        hint.setStyleSheet(f"color:{TEXT_SEC}; font-size:12px; background:transparent;")
        outer.addWidget(hint)

        # ── Tag picker: filter box + dropdown-style results + pills ────────────
        tag_group = QGroupBox("Tags to Apply")
        tg = QVBoxLayout(tag_group)
        filter_row = QHBoxLayout()
        self.tag_filter_edit = QLineEdit()
        self.tag_filter_edit.setPlaceholderText("Type to filter tags (e.g. \"color\")…")
        self.tag_filter_edit.textChanged.connect(self._refresh_tag_results)
        filter_row.addWidget(self.tag_filter_edit)
        tg.addLayout(filter_row)

        self.tag_results = QListWidget()
        self.tag_results.setFixedHeight(90)
        self.tag_results.itemClicked.connect(self._on_tag_result_clicked)
        tg.addWidget(self.tag_results)

        tg.addWidget(QLabel("Selected:"))
        self.pill_bar = TagPillBar()
        self.pill_bar.tag_removed.connect(self._remove_tag_from_selection)
        pill_scroll = QScrollArea(); pill_scroll.setWidgetResizable(True)
        pill_scroll.setFixedHeight(56)
        pill_scroll.setWidget(self.pill_bar)
        tg.addWidget(pill_scroll)
        outer.addWidget(tag_group)

        # ── File picker: search + zoom + thumbnail grid ─────────────────────────
        file_group = QGroupBox("Files")
        fg = QVBoxLayout(file_group)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText("Filter by filename…")
        self.search_edit.textChanged.connect(self._refresh_files)
        search_row.addWidget(self.search_edit)
        search_row.addStretch()
        search_row.addWidget(QLabel("Size:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(self.MIN_ZOOM, self.MAX_ZOOM)
        self.zoom_slider.setValue(self.DEFAULT_ZOOM)
        self.zoom_slider.setFixedWidth(120)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        search_row.addWidget(self.zoom_slider)
        fg.addLayout(search_row)

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.grid_container = QWidget()
        self.grid = FlowGridLayout(self.grid_container, h_spacing=8, v_spacing=8, margin=10)
        self.scroll.setWidget(self.grid_container)
        fg.addWidget(self.scroll, 1)

        select_row = QHBoxLayout()
        all_btn = QPushButton("Check All"); all_btn.clicked.connect(lambda: self._set_all_checked(True))
        select_row.addWidget(all_btn)
        none_btn = QPushButton("Uncheck All"); none_btn.clicked.connect(lambda: self._set_all_checked(False))
        select_row.addWidget(none_btn)
        select_row.addStretch()
        self.count_lbl = QLabel("")
        self.count_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        select_row.addWidget(self.count_lbl)
        fg.addLayout(select_row)
        outer.addWidget(file_group, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        apply_btn = QPushButton("Apply Changes"); apply_btn.setObjectName("accent")
        apply_btn.clicked.connect(self._apply_changes)
        btn_row.addWidget(apply_btn)
        outer.addLayout(btn_row)

    # ── Tag selection (pills) ────────────────────────────────────────────────

    def _load_vocab(self):
        self._all_vocab = self.db.get_all_tag_categories()
        self._refresh_tag_results()

    def _refresh_tag_results(self):
        query = self.tag_filter_edit.text().strip().lower()
        self.tag_results.clear()
        for cat, sub in self._all_vocab:
            label = f"{cat}: {sub}"
            if query and query not in label.lower():
                continue
            if (cat, sub) in self._selected_tags:
                continue   # already picked — don't clutter the results
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, (cat, sub))
            self.tag_results.addItem(item)

    def _on_tag_result_clicked(self, item: QListWidgetItem):
        tag = item.data(Qt.UserRole)
        self._add_tag_to_selection(tag)

    def _add_tag_to_selection(self, tag: Tuple[str,str]):
        if tag in self._selected_tags: return
        self._selected_tags.append(tag)
        self.pill_bar.set_tags(self._selected_tags)
        self._refresh_tag_results()

    def _remove_tag_from_selection(self, tag: Tuple[str,str]):
        if tag in self._selected_tags:
            self._selected_tags.remove(tag)
            self.pill_bar.set_tags(self._selected_tags)
            self._refresh_tag_results()

    # ── File grid ─────────────────────────────────────────────────────────────

    def _refresh_files(self):
        if self._fetcher and self._fetcher.isRunning():
            self._fetcher.cancel(); self._fetcher.wait(200)

        self.grid.clear()
        self._cards.clear()
        self._checked.clear()
        name = self.search_edit.text()
        self._rows = self.db.search_files_flat(name=name)

        for row in self._rows:
            card = PickerThumb(row['id'], row['filename'], size=self.zoom_slider.value())
            card.toggled.connect(self._on_card_toggled)
            self.grid.addWidget(card)
            self._cards[row['id']] = card

        self._update_count()
        if len(self._rows) >= 2000:
            self.count_lbl.setText(
                f"{self.count_lbl.text()}  (showing first 2000 — refine search)")

        if self._rows:
            self._fetcher = ThumbFetcher(self.db.lib_path, [r['id'] for r in self._rows])
            self._fetcher.result.connect(self._on_thumb_result, Qt.QueuedConnection)
            self._fetcher.start()

    def _on_thumb_result(self, fid: int, qimg, _frame_count: int):
        card = self._cards.get(fid)
        if card:
            card.set_thumbnail(qimg)

    def _on_card_toggled(self, fid: int, checked: bool):
        if checked: self._checked.add(fid)
        else: self._checked.discard(fid)
        self._update_count()

    def _on_zoom_changed(self, value: int):
        for card in self._cards.values():
            card.set_size(value)

    def _update_count(self):
        self.count_lbl.setText(f"{len(self._checked)} of {len(self._rows)} checked")

    def _set_all_checked(self, checked: bool):
        for fid, card in self._cards.items():
            card.set_checked(checked)
        self._checked = set(self._cards.keys()) if checked else set()
        self._update_count()

    # ── Apply ─────────────────────────────────────────────────────────────────

    def _apply_changes(self):
        if not self._selected_tags:
            QMessageBox.information(self, "No Tags Selected",
                "Pick at least one tag to apply (type to filter, then click a result).")
            return
        if not self._checked:
            QMessageBox.information(self, "No Files Selected",
                "Check at least one file to apply tags to.")
            return

        file_ids = list(self._checked)
        for cat, sub in self._selected_tags:
            self.db.bulk_assign_tag(file_ids, cat, sub)

        tag_list = ", ".join(f"{c}: {s}" for c,s in self._selected_tags)
        QMessageBox.information(self, "Applied",
            f"Added {len(self._selected_tags)} tag(s) — {tag_list} — "
            f"to {len(file_ids)} file(s).\nExisting tags on those files were kept.")

# ── Tag manager dialog (bulk vocabulary builder + rename/merge/delete) ───────

class TagManagerDialog(QDialog):
    """
    Bulk tag vocabulary management — separate from the per-item Tag Editor.
    Left panel: categories (add / rename / delete).
    Right panel: subcategories under the selected category — bulk-add many
    at once via a multi-line text box (one per line), plus rename/merge/
    delete for individual subcategories.
    Bulk file assignment lives in a separate window (BulkTagAssignDialog),
    opened via the "Assign to Files…" button here.
    This window never touches file selection state — it only edits the
    tag vocabulary and, for rename/merge, the tags already assigned to files.
    """

    def __init__(self, db: LibraryDB, parent=None):
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Tag Manager")
        self.setMinimumSize(760, 520)
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        self._current_category: Optional[str] = None
        self._build_ui()
        self._refresh_categories()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16,16,16,16); outer.setSpacing(10)

        title = QLabel("Tag Manager")
        title.setStyleSheet(f"color:{TEXT_PRI}; font-size:16px; font-weight:bold; background:transparent;")
        outer.addWidget(title)
        sub = QLabel("Build your tag vocabulary here, then assign tags to files in bulk.")
        sub.setStyleSheet(f"color:{TEXT_SEC}; font-size:12px; background:transparent;")
        outer.addWidget(sub)

        split = QSplitter(Qt.Horizontal)

        # ── Left: categories ─────────────────────────────────────────────────
        left = QGroupBox("Categories")
        ll = QVBoxLayout(left)
        self.cat_list = QListWidget()
        self.cat_list.currentItemChanged.connect(self._on_category_selected)
        ll.addWidget(self.cat_list)

        new_cat_row = QHBoxLayout()
        self.new_cat_edit = QLineEdit(); self.new_cat_edit.setPlaceholderText("New category name")
        self.new_cat_edit.returnPressed.connect(self._add_category)
        # Suggest existing category names while typing — helps catch
        # near-duplicates (e.g. typing "Vehicel" when "Vehicle" exists).
        self._cat_completer = QCompleter([], self.new_cat_edit)
        self._cat_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._cat_completer.setFilterMode(Qt.MatchStartsWith)
        self.new_cat_edit.setCompleter(self._cat_completer)
        new_cat_row.addWidget(self.new_cat_edit)
        add_cat_btn = QPushButton("Add"); add_cat_btn.setObjectName("accent")
        add_cat_btn.clicked.connect(self._add_category)
        new_cat_row.addWidget(add_cat_btn)
        ll.addLayout(new_cat_row)

        cat_actions = QHBoxLayout()
        ren_cat_btn = QPushButton("Rename…"); ren_cat_btn.clicked.connect(self._rename_category)
        cat_actions.addWidget(ren_cat_btn)
        del_cat_btn = QPushButton("Delete"); del_cat_btn.setObjectName("danger")
        del_cat_btn.clicked.connect(self._delete_category)
        cat_actions.addWidget(del_cat_btn)
        ll.addLayout(cat_actions)
        split.addWidget(left)

        # ── Right: subcategories under selected category ───────────────────────
        right = QGroupBox("Subcategories")
        rl = QVBoxLayout(right)
        self.sub_list = QListWidget()
        self.sub_list.setSelectionMode(QListWidget.ExtendedSelection)
        rl.addWidget(self.sub_list)

        rl.addWidget(QLabel("Bulk add (one per line):"))
        self.bulk_add_edit = QTextEdit()
        self.bulk_add_edit.setPlaceholderText("cars\ntrains\nplanes\nboats")
        self.bulk_add_edit.setFixedHeight(80)
        rl.addWidget(self.bulk_add_edit)
        bulk_add_btn = QPushButton("Add All"); bulk_add_btn.setObjectName("accent")
        bulk_add_btn.clicked.connect(self._bulk_add_subcategories)
        rl.addWidget(bulk_add_btn)

        sub_actions = QHBoxLayout()
        ren_sub_btn = QPushButton("Rename…"); ren_sub_btn.clicked.connect(self._rename_subcategory)
        sub_actions.addWidget(ren_sub_btn)
        merge_sub_btn = QPushButton("Merge Into…"); merge_sub_btn.clicked.connect(self._merge_subcategory)
        sub_actions.addWidget(merge_sub_btn)
        del_sub_btn = QPushButton("Delete"); del_sub_btn.setObjectName("danger")
        del_sub_btn.clicked.connect(self._delete_subcategories)
        sub_actions.addWidget(del_sub_btn)
        rl.addLayout(sub_actions)

        assign_btn = QPushButton("📌  Assign to Files…")
        assign_btn.clicked.connect(self._open_bulk_assign)
        rl.addWidget(assign_btn)

        split.addWidget(right)
        split.setSizes([260, 500])
        outer.addWidget(split, 1)

        backup_row = QHBoxLayout()
        backup_row.addWidget(QLabel("Backup:"))
        export_full_btn = QPushButton("Export Tags (CSV)…")
        export_full_btn.clicked.connect(self._export_full_csv)
        backup_row.addWidget(export_full_btn)
        export_vocab_btn = QPushButton("Export Vocabulary Only (CSV)…")
        export_vocab_btn.clicked.connect(self._export_vocab_csv)
        backup_row.addWidget(export_vocab_btn)
        import_btn = QPushButton("Import CSV…")
        import_btn.clicked.connect(self._import_csv)
        backup_row.addWidget(import_btn)
        backup_row.addStretch()
        outer.addLayout(backup_row)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.accept); btns.accepted.connect(self.accept)
        outer.addWidget(btns)

    # ── Category list ────────────────────────────────────────────────────────

    def _refresh_categories(self, select: Optional[str] = None):
        self.cat_list.blockSignals(True)
        self.cat_list.clear()
        cats = self.db.get_all_categories()
        for cat in cats:
            self.cat_list.addItem(cat)
        self.cat_list.blockSignals(False)
        self._cat_completer.setModel(QStringListModel(cats, self._cat_completer))
        if select and select in cats:
            items = self.cat_list.findItems(select, Qt.MatchExactly)
            if items: self.cat_list.setCurrentItem(items[0])
        elif cats:
            self.cat_list.setCurrentRow(0)
        else:
            self._current_category = None
            self._refresh_subcategories()

    def _on_category_selected(self, current, previous):
        self._current_category = current.text() if current else None
        self._refresh_subcategories()

    def _add_category(self):
        name = self.new_cat_edit.text().strip()
        if not name:
            return
        existing = self.db.get_all_categories()
        if name in existing:
            QMessageBox.information(self, "Already Exists",
                f"Category '{name}' already exists.")
            self.new_cat_edit.clear()
            return
        # A category only becomes visible once it has at least one
        # subcategory (tag_vocab requires both fields) — prompt for the
        # first one immediately so "create vehicle" feels like one step.
        sub, ok = QInputDialog.getText(self, "First Subcategory",
            f"Category '{name}' needs at least one subcategory to start.\n"
            "Enter the first one (you can bulk-add more after):")
        if not ok or not sub.strip():
            return
        self.db.add_tag_to_vocab(name, sub.strip())
        self.new_cat_edit.clear()
        self._refresh_categories(select=name)

    def _rename_category(self):
        if not self._current_category:
            QMessageBox.information(self, "No Selection", "Select a category first.")
            return
        new_name, ok = QInputDialog.getText(self, "Rename Category",
            "New name:", text=self._current_category)
        if not ok or not new_name.strip() or new_name.strip() == self._current_category:
            return
        self.db.rename_category(self._current_category, new_name.strip())
        self._refresh_categories(select=new_name.strip())

    def _delete_category(self):
        if not self._current_category:
            QMessageBox.information(self, "No Selection", "Select a category first.")
            return
        cat = self._current_category
        subs = self.db.get_subcategories(cat)
        usage = sum(self.db.get_tag_usage_count(cat, s) for s in subs)
        msg = f"Delete category '{cat}' and all {len(subs)} subcategor{'y' if len(subs)==1 else 'ies'}?"
        if usage:
            msg += f"\n\nThis will also remove it from {usage} file assignment(s)."
        if QMessageBox.question(self, "Delete Category", msg,
                                QMessageBox.Yes|QMessageBox.No) != QMessageBox.Yes:
            return
        self.db.delete_category_globally(cat)
        self._refresh_categories()

    # ── Subcategory list ─────────────────────────────────────────────────────

    def _refresh_subcategories(self):
        self.sub_list.clear()
        if not self._current_category:
            return
        for sub in self.db.get_subcategories(self._current_category):
            self.sub_list.addItem(sub)

    def _bulk_add_subcategories(self):
        if not self._current_category:
            QMessageBox.information(self, "No Category",
                "Select or create a category first.")
            return
        lines = [l.strip() for l in self.bulk_add_edit.toPlainText().splitlines()]
        lines = [l for l in lines if l]
        if not lines:
            return
        added = self.db.add_subcategories_bulk(self._current_category, lines)
        self.bulk_add_edit.clear()
        self._refresh_subcategories()
        skipped = len(lines) - added
        msg = f"Added {added} subcategor{'y' if added==1 else 'ies'}."
        if skipped:
            msg += f"\n{skipped} already existed and were skipped."
        self.status_flash(msg)

    def status_flash(self, msg: str):
        # Lightweight non-blocking confirmation — avoids a modal popup for
        # every bulk-add, which would break the flow of building a vocabulary.
        self.setWindowTitle(f"Tag Manager — {msg}")
        QTimer.singleShot(2500, lambda: self.setWindowTitle("Tag Manager"))

    def _selected_subs(self) -> List[str]:
        return [i.text() for i in self.sub_list.selectedItems()]

    def _rename_subcategory(self):
        subs = self._selected_subs()
        if len(subs) != 1:
            QMessageBox.information(self, "Select One",
                "Select exactly one subcategory to rename.")
            return
        old = subs[0]
        new, ok = QInputDialog.getText(self, "Rename Subcategory", "New name:", text=old)
        if not ok or not new.strip() or new.strip() == old:
            return
        self.db.rename_subcategory(self._current_category, old, new.strip())
        self._refresh_subcategories()

    def _merge_subcategory(self):
        subs = self._selected_subs()
        if len(subs) != 1:
            QMessageBox.information(self, "Select One",
                "Select exactly one subcategory to merge into another.")
            return
        source = subs[0]
        target, ok = QInputDialog.getText(self, "Merge Into",
            f"Merge '{self._current_category}: {source}' into which existing tag?\n"
            "Format: category: subcategory")
        if not ok or ':' not in target:
            return
        tc, ts = target.split(':', 1)
        tc, ts = tc.strip(), ts.strip()
        if not tc or not ts:
            return
        usage = self.db.get_tag_usage_count(self._current_category, source)
        msg = f"Merge '{self._current_category}: {source}' into '{tc}: {ts}'?"
        if usage:
            msg += f"\n\n{usage} file(s) currently tagged with the source will be re-tagged."
        msg += "\n\nThe source tag will be removed. This cannot be undone."
        if QMessageBox.question(self, "Merge Tags", msg,
                                QMessageBox.Yes|QMessageBox.No) != QMessageBox.Yes:
            return
        self.db.merge_tags(self._current_category, source, tc, ts)
        self._refresh_categories(select=self._current_category if self._current_category==tc else tc)

    def _delete_subcategories(self):
        subs = self._selected_subs()
        if not subs:
            QMessageBox.information(self, "No Selection", "Select at least one subcategory.")
            return
        usage = sum(self.db.get_tag_usage_count(self._current_category, s) for s in subs)
        msg = f"Delete {len(subs)} subcategor{'y' if len(subs)==1 else 'ies'} from '{self._current_category}'?"
        if usage:
            msg += f"\n\nThis will also remove it from {usage} file assignment(s)."
        if QMessageBox.question(self, "Delete Subcategories", msg,
                                QMessageBox.Yes|QMessageBox.No) != QMessageBox.Yes:
            return
        for s in subs:
            self.db.delete_tag_globally(self._current_category, s)
        self._refresh_subcategories()
        # If that was the last subcategory, the category itself disappears
        # from tag_vocab too (a category needs >=1 subcategory to exist)
        if not self.db.get_subcategories(self._current_category):
            self._refresh_categories()

    def _open_bulk_assign(self):
        subs = self._selected_subs()
        preset_tag = None
        if len(subs) == 1 and self._current_category:
            preset_tag = (self._current_category, subs[0])
        dlg = BulkTagAssignDialog(self.db, preset_tag=preset_tag, parent=self)
        dlg.exec_()
        self._refresh_categories(select=self._current_category)   # counts may have changed

    # ── Backup: export / import ──────────────────────────────────────────────

    def _export_full_csv(self):
        rows = self.db.get_export_rows()
        if not rows:
            QMessageBox.information(self, "Nothing to Export", "No files in the library yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Tags (Full)", "tags_export.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['rel_path','filename','category','subcategory','status'])
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            log.exception("CSV export failed")
            QMessageBox.critical(self, "Export Failed", str(e))
            return
        missing = sum(1 for r in rows if r['status'] == 'MISSING')
        msg = f"Exported {len(rows)} row(s) to:\n{path}"
        if missing:
            msg += f"\n\n{missing} row(s) reference files no longer found on disk (marked MISSING)."
        QMessageBox.information(self, "Export Complete", msg)

    def _export_vocab_csv(self):
        rows = self.db.get_vocab_export_rows()
        if not rows:
            QMessageBox.information(self, "Nothing to Export", "No tags in the vocabulary yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Vocabulary", "tag_vocabulary.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['category','subcategory'])
                writer.writeheader()
                writer.writerows(rows)
        except Exception as e:
            log.exception("Vocabulary CSV export failed")
            QMessageBox.critical(self, "Export Failed", str(e))
            return
        QMessageBox.information(self, "Export Complete",
            f"Exported {len(rows)} tag(s) to:\n{path}")

    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Tags CSV", "", "CSV Files (*.csv)")
        if not path: return
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = set(reader.fieldnames or [])
                rows = list(reader)
        except Exception as e:
            log.exception("CSV import failed to read file")
            QMessageBox.critical(self, "Import Failed", str(e))
            return

        if not rows:
            QMessageBox.information(self, "Empty File", "The selected CSV has no rows.")
            return

        # Detect which export format this is by its columns.
        if 'rel_path' in fieldnames:
            applied, missing_count, missing_paths = self.db.import_tag_rows(rows)
            msg = f"Applied {applied} tag assignment(s)."
            if missing_count:
                preview = "\n".join(missing_paths[:10])
                more = f"\n… and {missing_count-10} more" if missing_count > 10 else ""
                msg += (f"\n\n{missing_count} row(s) referenced files not found in this "
                       f"library and were skipped:\n{preview}{more}")
            QMessageBox.information(self, "Import Complete", msg)
        elif 'category' in fieldnames and 'subcategory' in fieldnames:
            added = self.db.import_vocab_rows(rows)
            QMessageBox.information(self, "Import Complete",
                f"Added {added} new tag(s) to the vocabulary.\n"
                f"{len(rows)-added} already existed and were skipped.")
        else:
            QMessageBox.warning(self, "Unrecognized Format",
                "This CSV doesn't match either export format "
                "(expected 'rel_path,filename,category,subcategory,status' "
                "or 'category,subcategory').")
            return

        self._refresh_categories(select=self._current_category)


# ── Tag editor dialog (per-item — coexists with Tag Manager) ─────────────────

class TagEditorDialog(QDialog):
    def __init__(self, db: LibraryDB, file_ids: List[int], parent=None):
        super().__init__(parent)
        self.db=db; self.file_ids=file_ids
        self.setWindowTitle("Edit Tags")
        self.setMinimumSize(600,500)
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        self._build()

    def _build(self):
        layout=QHBoxLayout(self); layout.setSpacing(12)
        left=QGroupBox("Available Tags"); ll=QVBoxLayout(left)
        search=QLineEdit(); search.setPlaceholderText("Filter tags…")
        search.textChanged.connect(self._filter); ll.addWidget(search)
        self.avail_tree=QTreeWidget(); self.avail_tree.setHeaderHidden(True)
        self.avail_tree.setAlternatingRowColors(True); ll.addWidget(self.avail_tree)
        new_row=QHBoxLayout()
        self.new_cat=QLineEdit(); self.new_cat.setPlaceholderText("Category")
        self.new_sub=QLineEdit(); self.new_sub.setPlaceholderText("Subcategory")
        add_btn=QPushButton("Add"); add_btn.setObjectName("accent")
        add_btn.clicked.connect(self._add_new)
        new_row.addWidget(self.new_cat); new_row.addWidget(QLabel(":"))
        new_row.addWidget(self.new_sub); new_row.addWidget(add_btn)
        ll.addLayout(new_row); layout.addWidget(left)
        mid=QVBoxLayout(); mid.addStretch()
        ab=QPushButton("→ Assign"); ab.setObjectName("accent")
        ab.clicked.connect(self._assign)
        rb=QPushButton("← Remove"); rb.clicked.connect(self._remove)
        db_btn=QPushButton("🗑 Delete Tag"); db_btn.setObjectName("danger")
        db_btn.clicked.connect(self._del_tag)
        mid.addWidget(ab); mid.addSpacing(6); mid.addWidget(rb)
        mid.addSpacing(20); mid.addWidget(db_btn); mid.addStretch()
        layout.addLayout(mid)
        right=QGroupBox(f"Assigned Tags ({len(self.file_ids)} file(s))")
        rl=QVBoxLayout(right); self.assigned=QListWidget(); rl.addWidget(self.assigned)
        layout.addWidget(right)
        outer=QVBoxLayout(); outer.addLayout(layout)
        btns=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save); btns.rejected.connect(self.reject)
        outer.addWidget(btns); self.setLayout(outer)
        self._refresh_avail(); self._refresh_assigned()

    def _refresh_avail(self):
        self.avail_tree.clear(); cats={}
        for cat,sub in self.db.get_all_tag_categories():
            if cat not in cats:
                item=QTreeWidgetItem([cat]); item.setForeground(0,QColor(ACCENT))
                self.avail_tree.addTopLevelItem(item); cats[cat]=item
            child=QTreeWidgetItem([sub]); child.setData(0,Qt.UserRole,(cat,sub))
            cats[cat].addChild(child)
        self.avail_tree.expandAll()

    def _filter(self,text):
        for i in range(self.avail_tree.topLevelItemCount()):
            top=self.avail_tree.topLevelItem(i); any_vis=False
            for j in range(top.childCount()):
                child=top.child(j)
                vis=text.lower() in child.text(0).lower() or text.lower() in top.text(0).lower()
                child.setHidden(not vis)
                if vis: any_vis=True
            top.setHidden(not any_vis)

    def _refresh_assigned(self):
        self.assigned.clear()
        if len(self.file_ids)==1:
            for cat,sub in self.db.get_tags(self.file_ids[0]):
                item=QListWidgetItem(f"{cat}: {sub}")
                item.setData(Qt.UserRole,(cat,sub)); self.assigned.addItem(item)

    def _sel_avail(self):
        sel=self.avail_tree.currentItem()
        return sel.data(0,Qt.UserRole) if sel else None

    def _assign(self):
        tag=self._sel_avail()
        if not tag: return
        cat,sub=tag; label=f"{cat}: {sub}"
        for i in range(self.assigned.count()):
            if self.assigned.item(i).text()==label: return
        item=QListWidgetItem(label); item.setData(Qt.UserRole,(cat,sub))
        self.assigned.addItem(item)

    def _remove(self):
        for item in self.assigned.selectedItems():
            self.assigned.takeItem(self.assigned.row(item))

    def _add_new(self):
        cat=self.new_cat.text().strip(); sub=self.new_sub.text().strip()
        if not cat or not sub:
            QMessageBox.warning(self,"Missing fields","Enter category and subcategory."); return
        # Register in the vocabulary directly — no longer needs the old
        # workaround of assigning to a file just to make the tag "exist".
        self.db.add_tag_to_vocab(cat, sub)
        self._refresh_avail()
        label=f"{cat}: {sub}"
        for i in range(self.assigned.count()):
            if self.assigned.item(i).text()==label: return
        item=QListWidgetItem(label); item.setData(Qt.UserRole,(cat,sub))
        self.assigned.addItem(item); self.new_cat.clear(); self.new_sub.clear()

    def _del_tag(self):
        tag=self._sel_avail()
        if not tag: return
        cat,sub=tag
        if QMessageBox.question(self,"Delete Tag",
            f"Delete '{cat}: {sub}' from ALL files?",
            QMessageBox.Yes|QMessageBox.No)==QMessageBox.Yes:
            self.db.delete_tag_globally(cat, sub)
            self._refresh_avail(); self._refresh_assigned()

    def _save(self):
        if len(self.file_ids)==1:
            tags=[self.assigned.item(i).data(Qt.UserRole) for i in range(self.assigned.count())]
            self.db.set_tags(self.file_ids[0],tags)
        else:
            tags_add=[self.assigned.item(i).data(Qt.UserRole) for i in range(self.assigned.count())]
            for fid in self.file_ids:
                self.db.set_tags(fid,list(set(self.db.get_tags(fid)+tags_add)))
        self.accept()

# ── File details dialog ───────────────────────────────────────────────────────

class FileDetailsDialog(QDialog):
    def __init__(self, db: LibraryDB, fid: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Details"); self.setMinimumSize(450,400)
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        row=db.get_file_by_id(fid); tags=db.get_tags(fid)
        layout=QFormLayout(self); layout.setSpacing(10)
        layout.setContentsMargins(20,20,20,20)
        def field(lbl,val):
            l=QLabel(str(val or "—")); l.setWordWrap(True)
            layout.addRow(QLabel(lbl),l)
        field("Filename:",row['filename']); field("Type:",row['file_type'].capitalize())
        field("Relative Path:",row['rel_path']); field("Absolute Path:",row['abs_path'])
        if row['file_size']:
            sz=row['file_size']
            field("File Size:", f"{sz/1048576:.1f} MB" if sz>1e6 else
                               f"{sz/1024:.1f} KB" if sz>1000 else f"{sz} B")
        if row['width'] and row['height']:
            field("Dimensions:",f"{row['width']} × {row['height']} px")
        if row['duration']:
            m,s=divmod(int(row['duration']),60); field("Duration:",f"{m}m {s}s")
        field("Date Added:",row['date_added']); field("Date Modified:",row['date_modified'])
        field("MD5:",row['checksum'])
        field("Tags:", ", ".join(f"{c}: {s}" for c,s in tags) if tags else "None")
        btns=QDialogButtonBox(QDialogButtonBox.Close); btns.rejected.connect(self.reject)
        layout.addRow(btns)

# ── Library information dialog ────────────────────────────────────────────────

def _format_bytes(n: float) -> str:
    if not n: return "0 B"
    if n >= 1024**3: return f"{n/1024**3:.2f} GB"
    if n >= 1024**2: return f"{n/1024**2:.1f} MB"
    if n >= 1024:     return f"{n/1024:.1f} KB"
    return f"{int(n)} B"

class LibraryInfoDialog(QDialog):
    """
    Read-only overview of the current library: file counts by type, total
    and average size, tag/vocabulary usage, ratings, missing files, date
    range, tracked source folders, and the largest files on disk.
    """

    def __init__(self, db: LibraryDB, lib_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Library Information")
        self.setMinimumSize(480, 560)
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        stats = db.get_library_stats()
        self._build_ui(stats, lib_path)

    def _build_ui(self, s: dict, lib_path: str):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header.setStyleSheet(f"background:{PANEL_BG}; border-bottom:2px solid {BORDER};")
        hl = QVBoxLayout(header)
        hl.setContentsMargins(20, 14, 20, 14)
        hl.setSpacing(4)
        title = QLabel("Library Information")
        title.setStyleSheet(f"color:{TEXT_PRI}; font-size:16px; font-weight:bold; background:transparent;")
        hl.addWidget(title)
        path_lbl = QLabel(lib_path)
        path_lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px; background:transparent;")
        path_lbl.setWordWrap(True)
        hl.addWidget(path_lbl)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(20, 14, 20, 14)
        cl.setSpacing(14)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        def section(heading: str) -> QFormLayout:
            box = QGroupBox(heading)
            form = QFormLayout(box)
            form.setSpacing(6)
            cl.addWidget(box)
            return form

        def row(form, label, value):
            l = QLabel(str(value))
            l.setStyleSheet(f"color:{TEXT_PRI}; background:transparent;")
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
            form.addRow(lbl, l)

        f1 = section("Contents")
        row(f1, "Images:",  f"{s['images']:,}")
        row(f1, "Videos:",  f"{s['videos']:,}")
        row(f1, "Zips:",    f"{s['zips']:,}")
        row(f1, "Folders:", f"{s['folders']:,}")
        row(f1, "Total files:", f"{s['total_files']:,}")

        f2 = section("Storage")
        row(f2, "Total size:",   _format_bytes(s['total_size']))
        row(f2, "Average file size:", _format_bytes(s['avg_size']))
        row(f2, "Library database file:", _format_bytes(s['db_file_size']))

        f3 = section("Tags & Ratings")
        row(f3, "Categories:", f"{s['category_count']:,}")
        row(f3, "Tags (total):", f"{s['tag_count']:,}")
        untagged_pct = (s['untagged_files'] / s['total_files'] * 100) if s['total_files'] else 0
        row(f3, "Untagged files:", f"{s['untagged_files']:,} ({untagged_pct:.0f}%)")
        row(f3, "Files with stars:", f"{s['starred_files']:,}")
        row(f3, "Files with hearts:", f"{s['hearted_files']:,}")

        f4 = section("Library Health")
        row(f4, "Deleted (missing) entries:", f"{s['deleted_files']:,}")
        if s['deleted_files']:
            hint = QLabel("Run Compact Database to permanently remove these and reclaim space.")
            hint.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px; background:transparent;")
            hint.setWordWrap(True)
            f4.addRow(hint)
        row(f4, "Tracked source folders:", f"{s['source_count']:,}")

        oldest = (s['oldest_date'] or '')[:10] or "—"
        newest = (s['newest_date'] or '')[:10] or "—"
        f5 = section("Date Range (by modified date)")
        row(f5, "Oldest file:", oldest)
        row(f5, "Newest file:", newest)

        if s['largest_files']:
            f6 = section("Largest Files (click to open)")
            for i, r in enumerate(s['largest_files'], start=1):
                self._add_largest_file_row(f6, i, r)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.accept); btns.accepted.connect(self.accept)
        outer.addWidget(btns)

    def _add_largest_file_row(self, form: QFormLayout, rank: int, file_row):
        """One ranked, clickable row in the Largest Files section. Clicking
        opens the file in its OS default application — same behavior as
        double-clicking it in the main gallery — or, if the file is no
        longer on disk, reports that clearly instead."""
        rank_lbl = QLabel(f"{rank}.")
        rank_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")

        btn = QPushButton(f"{file_row['filename']}  —  {_format_bytes(file_row['file_size'])}")
        btn.setFlat(True)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton{{color:{ACCENT2}; background:transparent; border:none; "
            f"text-align:left; text-decoration:underline; padding:2px 0;}}"
            f"QPushButton:hover{{color:{ACCENT};}}"
        )
        abs_path = file_row['abs_path']
        btn.clicked.connect(lambda _, p=abs_path, n=file_row['filename']:
                            self._open_largest_file(p, n))
        form.addRow(rank_lbl, btn)

    def _open_largest_file(self, abs_path: str, filename: str):
        if not abs_path or not os.path.exists(abs_path):
            QMessageBox.warning(self, "File Not Found",
                f"'{filename}' could not be found on disk:\n{abs_path or '(no path recorded)'}")
            return
        try:
            os.startfile(abs_path)
        except AttributeError:
            subprocess.Popen(['xdg-open', abs_path])
        except Exception as e:
            QMessageBox.warning(self, "Cannot Open", str(e))

# ── Zip export preview dialog ─────────────────────────────────────────────────

# ── Simple wrapping flow layout (used by ZipExportPreviewDialog) ─────────────

class FlowGridLayout(QLayout):
    """Left-to-right wrapping layout for a fixed-width scroll area content
    widget — lays out child widgets in rows, wrapping to a new row when
    the next item would overflow the available width."""

    def __init__(self, parent=None, h_spacing=8, v_spacing=8, margin=8):
        super().__init__(parent)
        self._items = []
        self._h_space = h_spacing
        self._v_space = v_spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item): self._items.append(item)
    def count(self): return len(self._items)
    def itemAt(self, i): return self._items[i] if 0<=i<len(self._items) else None
    def takeAt(self, i): return self._items.pop(i) if 0<=i<len(self._items) else None
    def hasHeightForWidth(self): return True
    def heightForWidth(self, w): return self._do_layout(QRect(0,0,w,0), True)

    def clear(self):
        """Remove and delete every widget currently in this layout —
        needed by any dialog that repopulates the grid on search/filter
        changes (e.g. BulkTagAssignDialog, TagPillBar)."""
        while self._items:
            item = self._items.pop()
            wid = item.widget()
            if wid is not None:
                wid.setParent(None)
                wid.deleteLater()

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)
    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        size = QSize()
        for item in self._items: size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left()+m.right(), m.top()+m.bottom())

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = eff.x(), eff.y(), 0
        for item in self._items:
            wid = item.widget()
            if wid is None or wid.isHidden(): continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_space
            if next_x - self._h_space > eff.right() and row_h > 0:
                x = eff.x(); y += row_h + self._v_space
                next_x = x + hint.width() + self._h_space
                row_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x,y), hint))
            x = next_x
            row_h = max(row_h, hint.height())
        return y + row_h - rect.y() + m.bottom()

class ZipPageThumb(QFrame):
    """
    One page thumbnail inside a zip's grid view. Two modes:
      'select' — export preview: click toggles a checkmark (Ctrl/Shift
                 modify selection like the main gallery). Used by
                 ZipExportPreviewDialog.
      'browse' — reader folder view: no checkmark at all, single click
                 does nothing special, double-click opens the reader.
                 Used by ZipFolderDialog.
    Selection-state logic (for 'select' mode) lives in the parent dialog —
    this widget only reports clicks/double-clicks and renders its state.

    Size is instance-configurable via set_size() so a zoom slider in the
    parent dialog can resize all thumbnails live without recreating them —
    the source QImage is kept so rescaling stays sharp at any zoom level.
    """

    clicked        = pyqtSignal(int, object)   # index, QMouseEvent
    double_clicked = pyqtSignal(int)           # index

    DEFAULT_SIZE = 150

    def __init__(self, index: int, filename: str, mode: str = 'select',
                 size: int = DEFAULT_SIZE, parent=None):
        super().__init__(parent)
        self.index    = index
        self.filename = filename
        self.mode     = mode
        self.checked  = (mode == 'select')   # export mode starts all-checked
        self.size_px  = size
        self._source_qimg: Optional[QImage] = None
        self.setCursor(Qt.PointingHandCursor)
        self._build()
        self.set_size(size)

    def _build(self):
        layout = QVBoxLayout(self); layout.setContentsMargins(4,4,4,4); layout.setSpacing(3)
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet(f"background:{CARD_BG}; border-radius:4px;")
        self.thumb_label.setText("…")
        layout.addWidget(self.thumb_label)
        self.name_lbl = QLabel(self.filename)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px; background:transparent;")
        layout.addWidget(self.name_lbl)
        self._update_style()

    def set_size(self, size: int):
        """Live-resize this thumbnail (used by the zoom slider). Re-scales
        from the retained source QImage so quality doesn't degrade when
        zooming back up after zooming down."""
        self.size_px = size
        self.setFixedSize(size, size + 26)
        self.thumb_label.setFixedSize(size-8, size-8)
        fm = QFontMetrics(self.name_lbl.font())
        self.name_lbl.setText(fm.elidedText(self.filename, Qt.ElideMiddle, size-8))
        self._render_thumb()

    def set_thumbnail(self, qimg: Optional[QImage]):
        self._source_qimg = qimg
        self._load_failed = (qimg is None)
        self._render_thumb()

    def _render_thumb(self):
        if self._source_qimg is not None:
            pix = QPixmap.fromImage(self._source_qimg)
            pix = pix.scaled(self.size_px-8, self.size_px-8,
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.thumb_label.setPixmap(pix)
        else:
            self.thumb_label.setPixmap(QPixmap())
            self.thumb_label.setText("✕" if getattr(self, '_load_failed', False) else "…")

    def set_checked(self, checked: bool):
        self.checked = checked
        self._update_style()
        self.update()

    def _update_style(self):
        border = SEL_COL if (self.mode == 'select' and self.checked) else BORDER
        self.setStyleSheet(f"""
            ZipPageThumb {{
                background: {CARD_BG};
                border: 2px solid {border};
                border-radius: 6px;
            }}
            ZipPageThumb:hover {{
                border-color: {ACCENT2};
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index, event)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self.index)
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.mode != 'select' or not self.checked:
            return
        # Checkmark badge, top-right, matching the export-preview mockup
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = 20
        bx = self.width() - r - 4
        by = 4
        p.setBrush(QBrush(QColor(SUCCESS)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(bx, by, r, r)
        p.setPen(QPen(QColor('white'), 2))
        # simple checkmark
        p.drawLine(bx+4, by+10, bx+8, by+14)
        p.drawLine(bx+8, by+14, bx+16, by+5)
        p.end()


class ZipExportPreviewDialog(QDialog):
    """
    Preview + select which images inside a single zip get exported as PDF
    pages. Thumbnails stream in progressively via ZipThumbLoader so the
    dialog is responsive even for large zips. All images start checked.
    """

    MIN_ZOOM, MAX_ZOOM, DEFAULT_ZOOM = 80, 320, 150

    def __init__(self, zip_path: str, zip_filename: str, parent=None):
        super().__init__(parent)
        self.zip_path = zip_path
        self.setWindowTitle(zip_filename)
        self.setMinimumSize(900, 650)
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        # Enable the maximize button in the title bar (QDialog omits it by
        # default) alongside the normal close/minimize controls.
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)

        self._thumbs: Dict[int, ZipPageThumb] = {}
        self._paths:  Dict[int, str] = {}       # index -> extracted file path
        self._checked: set = set()
        self._last_click_idx: Optional[int] = None
        self._zoom: int = self.DEFAULT_ZOOM
        self.export_mode: Optional[str] = None   # 'all' | 'selected' | None (cancelled)

        # Enhancement state — 'off' by default, export-time only per spec
        self._enhance_level: str = 'off'          # 'mild'|'strong'|'custom'|'off'
        self._custom_sharpen: float = 100          # UnsharpMask percent
        self._custom_contrast: float = 1.0         # ImageEnhance.Contrast factor
        self._custom_noise: float = 0              # noise reduction percent (Custom only)

        # Large preview pane state — separate from the small grid thumbnails.
        # Toggled via the "Preview" button; shows one image at a time, full
        # size, with the current adjustment applied, navigable with </>.
        self._preview_visible: bool = False
        self._preview_index: int = 0               # position in sorted(self._paths.keys())
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(150)   # coalesce rapid slider drags
        self._preview_debounce.timeout.connect(self._update_preview)

        self.extract_dir = os.path.join(
            get_session_tmp_dir(), f"preview_{os.path.basename(zip_path)}")

        self._build_ui(zip_filename)
        self._start_loading()

    def _build_ui(self, zip_filename: str):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        title_row = QWidget()
        title_row.setStyleSheet(f"background:{PANEL_BG}; border-bottom:2px solid {BORDER};")
        tl = QHBoxLayout(title_row); tl.setContentsMargins(14,8,14,8)
        title = QLabel(zip_filename)
        title.setStyleSheet(f"color:{TEXT_PRI}; font-size:15px; font-weight:bold; background:transparent;")
        tl.addWidget(title)
        tl.addStretch()
        zoom_lbl = QLabel("Size:")
        zoom_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        tl.addWidget(zoom_lbl)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(self.MIN_ZOOM, self.MAX_ZOOM)
        self.zoom_slider.setValue(self._zoom)
        self.zoom_slider.setFixedWidth(140)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        tl.addWidget(self.zoom_slider)
        tl.addSpacing(12)
        self.preview_toggle_btn = QPushButton("👁 Preview")
        self.preview_toggle_btn.setCheckable(True)
        self.preview_toggle_btn.setMinimumWidth(ws(90))
        self.preview_toggle_btn.setToolTip(
            "Show a full-size preview pane with the current adjustment applied")
        self.preview_toggle_btn.clicked.connect(self._toggle_preview_pane)
        tl.addWidget(self.preview_toggle_btn)
        outer.addWidget(title_row)

        # ── Enhancement row: Mild / Strong / Custom / Off ───────────────────────
        enh_row = QWidget()
        enh_row.setStyleSheet(f"background:{PANEL_BG}; border-bottom:1px solid {BORDER};")
        el = QHBoxLayout(enh_row); el.setContentsMargins(14,6,14,6); el.setSpacing(8)

        enh_lbl = QLabel("Sharpen / Contrast / Noise:")
        enh_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        el.addWidget(enh_lbl)

        self.enh_group = QButtonGroup(self)
        self.enh_buttons = {}
        for key, label in [('mild','Mild'), ('strong','Strong'),
                           ('custom','Custom'), ('off','Off')]:
            btn = QPushButton(label); btn.setCheckable(True)
            btn.setMinimumWidth(ws(64))
            btn.clicked.connect(lambda _,k=key: self._set_enhance_level(k))
            self.enh_group.addButton(btn)
            self.enh_buttons[key] = btn
            el.addWidget(btn)
        self.enh_buttons['off'].setChecked(True)

        el.addSpacing(10)
        self.sharpen_lbl = QLabel("Sharpen:")
        self.sharpen_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        el.addWidget(self.sharpen_lbl)
        self.sharpen_slider = QSlider(Qt.Horizontal)
        smin, smax = ENHANCE_CUSTOM_RANGE['sharpen']
        self.sharpen_slider.setRange(int(smin), int(smax))
        self.sharpen_slider.setValue(int(self._custom_sharpen))
        self.sharpen_slider.setFixedWidth(110)
        self.sharpen_slider.valueChanged.connect(self._on_custom_sharpen_changed)
        el.addWidget(self.sharpen_slider)

        self.contrast_lbl = QLabel("Contrast:")
        self.contrast_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        el.addWidget(self.contrast_lbl)
        self.contrast_slider = QSlider(Qt.Horizontal)
        cmin, cmax = ENHANCE_CUSTOM_RANGE['contrast']
        self.contrast_slider.setRange(int(cmin*100), int(cmax*100))
        self.contrast_slider.setValue(int(self._custom_contrast*100))
        self.contrast_slider.setFixedWidth(110)
        self.contrast_slider.valueChanged.connect(self._on_custom_contrast_changed)
        el.addWidget(self.contrast_slider)

        self.noise_lbl = QLabel("Noise Reduction:")
        self.noise_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        el.addWidget(self.noise_lbl)
        self.noise_slider = QSlider(Qt.Horizontal)
        nmin, nmax = ENHANCE_CUSTOM_RANGE['noise']
        self.noise_slider.setRange(int(nmin), int(nmax))
        self.noise_slider.setValue(int(self._custom_noise))
        self.noise_slider.setFixedWidth(110)
        self.noise_slider.valueChanged.connect(self._on_custom_noise_changed)
        el.addWidget(self.noise_slider)

        el.addStretch()
        outer.addWidget(enh_row)
        self._set_custom_controls_visible(False)   # hidden until 'custom' chosen

        # ── Main area: thumbnail grid (left) + toggleable preview pane (right) ──
        self.main_split = QSplitter(Qt.Horizontal)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.grid = FlowGridLayout(self.container, h_spacing=10, v_spacing=10, margin=14)
        self.scroll.setWidget(self.container)
        self.main_split.addWidget(self.scroll)

        self.preview_pane = self._build_preview_pane()
        self.main_split.addWidget(self.preview_pane)
        self.preview_pane.setVisible(False)   # hidden until the Preview button is toggled on
        self.main_split.setSizes([600, 400])

        outer.addWidget(self.main_split, 1)

        bottom = QWidget()
        bottom.setStyleSheet(f"background:{PANEL_BG}; border-top:1px solid {BORDER};")
        brow = QHBoxLayout(bottom); brow.setContentsMargins(14,10,14,10)
        self.status_lbl = QLabel("Loading…")
        self.status_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        brow.addWidget(self.status_lbl)
        brow.addStretch()
        cancel_btn = QPushButton("Cancel"); cancel_btn.clicked.connect(self.reject)
        brow.addWidget(cancel_btn)
        all_btn = QPushButton("Export all"); all_btn.clicked.connect(self._export_all)
        brow.addWidget(all_btn)
        sel_btn = QPushButton("Export selected"); sel_btn.setObjectName("accent")
        sel_btn.clicked.connect(self._export_selected)
        brow.addWidget(sel_btn)
        outer.addWidget(bottom)

    def _build_preview_pane(self) -> QWidget:
        """
        Large, full-size preview pane shown to the right of the thumbnail
        grid when the Preview toggle is on. Shows one image at a time with
        the current enhancement applied (at full resolution, unlike the
        earlier small swatch), navigable with </> independent of which
        thumbnails are checked/unchecked for export.
        """
        pane = QWidget()
        pane.setStyleSheet(f"background:{DARK_BG}; border-left:1px solid {BORDER};")
        pl = QVBoxLayout(pane)
        pl.setContentsMargins(10, 10, 10, 10)
        pl.setSpacing(8)

        head = QHBoxLayout()
        self.preview_filename_lbl = QLabel("")
        self.preview_filename_lbl.setStyleSheet(f"color:{TEXT_PRI}; background:transparent; font-weight:bold;")
        fm = QFontMetrics(self.preview_filename_lbl.font())
        head.addWidget(self.preview_filename_lbl, 1)
        self.preview_index_lbl = QLabel("")
        self.preview_index_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        head.addWidget(self.preview_index_lbl)
        pl.addLayout(head)

        self.preview_image_label = QLabel()
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setStyleSheet(
            f"background:{CARD_BG}; border:1px solid {BORDER}; border-radius:4px;")
        self.preview_image_label.setMinimumSize(200, 200)
        pl.addWidget(self.preview_image_label, 1)

        nav_row = QHBoxLayout()
        nav_row.addStretch()
        self.preview_prev_btn = QPushButton("<")
        self.preview_prev_btn.setMinimumWidth(ws(40))
        self.preview_prev_btn.clicked.connect(self._preview_prev)
        nav_row.addWidget(self.preview_prev_btn)
        self.preview_next_btn = QPushButton(">")
        self.preview_next_btn.setMinimumWidth(ws(40))
        self.preview_next_btn.clicked.connect(self._preview_next)
        nav_row.addWidget(self.preview_next_btn)
        nav_row.addStretch()
        pl.addLayout(nav_row)

        return pane

    def _set_enhance_level(self, level: str):
        self._enhance_level = level
        self._set_custom_controls_visible(level == 'custom')
        self._preview_debounce.start()

    def _set_custom_controls_visible(self, visible: bool):
        for w in (self.sharpen_lbl, self.sharpen_slider,
                  self.contrast_lbl, self.contrast_slider,
                  self.noise_lbl, self.noise_slider):
            w.setVisible(visible)

    def _on_custom_sharpen_changed(self, value: int):
        self._custom_sharpen = float(value)
        self._preview_debounce.start()

    def _on_custom_contrast_changed(self, value: int):
        self._custom_contrast = value / 100.0
        self._preview_debounce.start()

    def _on_custom_noise_changed(self, value: int):
        self._custom_noise = float(value)
        self._preview_debounce.start()

    def get_enhance_params(self):
        """Returns (sharpen_percent, contrast_factor, noise_percent) for
        the chosen level."""
        return enhance_params_for_level(
            self._enhance_level, self._custom_sharpen,
            self._custom_contrast, self._custom_noise)

    # ── Large preview pane ────────────────────────────────────────────────────

    def _toggle_preview_pane(self):
        self._preview_visible = self.preview_toggle_btn.isChecked()
        self.preview_pane.setVisible(self._preview_visible)
        if self._preview_visible:
            self._update_preview()

    def _preview_prev(self):
        ordered = sorted(self._paths.keys())
        if not ordered: return
        self._preview_index = max(0, self._preview_index - 1)
        self._update_preview()

    def _preview_next(self):
        ordered = sorted(self._paths.keys())
        if not ordered: return
        self._preview_index = min(len(ordered) - 1, self._preview_index + 1)
        self._update_preview()

    def _current_preview_path(self) -> Optional[str]:
        ordered = sorted(self._paths.keys())
        if not ordered:
            return None
        self._preview_index = max(0, min(self._preview_index, len(ordered) - 1))
        idx = ordered[self._preview_index]
        return self._paths.get(idx)

    def _update_preview(self):
        """
        Re-renders the large preview pane with the current enhancement
        settings applied to the FULL-RESOLUTION source image (not a small
        downscaled sample) — only the on-screen pixmap is scaled to fit
        the pane, so sharpen/contrast/noise-reduction effects are actually
        visible rather than lost in a tiny swatch. Only does this work
        while the pane is actually visible; a toggle-on or navigation
        click triggers a render immediately, slider drags are debounced.
        """
        if not self._preview_visible:
            return

        ordered = sorted(self._paths.keys())
        path = self._current_preview_path()
        if not path or not os.path.exists(path):
            self.preview_image_label.setText("No image to preview.")
            self.preview_filename_lbl.setText("")
            self.preview_index_lbl.setText("")
            self.preview_prev_btn.setEnabled(False)
            self.preview_next_btn.setEnabled(False)
            return

        fname = os.path.basename(path).split('_', 1)[-1]
        fm = QFontMetrics(self.preview_filename_lbl.font())
        self.preview_filename_lbl.setText(
            fm.elidedText(fname, Qt.ElideMiddle, self.preview_pane.width() - 100))
        self.preview_index_lbl.setText(f"{self._preview_index+1}/{len(ordered)}")
        self.preview_prev_btn.setEnabled(self._preview_index > 0)
        self.preview_next_btn.setEnabled(self._preview_index < len(ordered) - 1)

        sharpen_pct, contrast_f, noise_pct = self.get_enhance_params()
        try:
            with Image.open(path) as pil_img:
                if pil_img.mode not in ('RGB', 'L'):
                    pil_img = pil_img.convert('RGB')
                pil_img = enhance_image(pil_img, sharpen_pct, contrast_f, noise_pct)
                qimg = self._pil_to_qimage(pil_img)
            avail = self.preview_image_label.size()
            pix = QPixmap.fromImage(qimg).scaled(
                max(1, avail.width()-8), max(1, avail.height()-8),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_image_label.setPixmap(pix)
        except Exception:
            log.exception("ZipExportPreviewDialog: preview render failed for %s", path)
            self.preview_image_label.setText("Could not load preview.")

    @staticmethod
    def _pil_to_qimage(img: Image.Image) -> QImage:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        data = img.tobytes('raw', 'RGB')
        qimg = QImage(data, img.width, img.height, img.width*3, QImage.Format_RGB888)
        return qimg.copy()   # detach from the temporary 'data' buffer

    def _on_zoom_changed(self, value: int):
        self._zoom = value
        for t in self._thumbs.values():
            t.set_size(value)

    def _start_loading(self):
        self.loader = ZipThumbLoader(self.zip_path, self.extract_dir)
        self.loader.thumb_ready.connect(self._on_thumb_ready, Qt.QueuedConnection)
        self.loader.finished_sig.connect(self._on_load_finished, Qt.QueuedConnection)
        self.loader.error_sig.connect(self._on_load_error, Qt.QueuedConnection)
        self.loader.start()

    def _on_thumb_ready(self, index: int, path: str, qimg):
        # We don't know the filename ahead of time for the label, so derive
        # it from the extracted path (format: NNNNN_originalname.ext)
        fname = os.path.basename(path).split('_', 1)[-1] if path else f"page {index+1}"
        thumb = ZipPageThumb(index, fname, size=self._zoom)
        thumb.clicked.connect(self._on_thumb_click)
        thumb.set_thumbnail(qimg)
        thumb.set_checked(True)
        self._thumbs[index] = thumb
        self._paths[index]  = path
        self._checked.add(index)
        self.grid.addWidget(thumb)
        self.status_lbl.setText(f"Loaded {len(self._thumbs)} image(s)…")
        if self._preview_visible and len(self._paths) == 1:
            # First image to arrive while the pane is already open —
            # nothing to preview until now, so render immediately.
            self._update_preview()

    def _on_load_finished(self, total: int):
        self.status_lbl.setText(
            f"Selected {len(self._checked)} of {total} files" if total
            else "No images found in this zip.")

    def _on_load_error(self, msg: str):
        QMessageBox.warning(self, "Error", f"Could not read zip:\n{msg}")

    def _on_thumb_click(self, index: int, event):
        mods = event.modifiers()
        if mods & Qt.ControlModifier:
            if index in self._checked: self._checked.discard(index)
            else: self._checked.add(index)
            self._last_click_idx = index
        elif mods & Qt.ShiftModifier and self._last_click_idx is not None:
            lo, hi = sorted((self._last_click_idx, index))
            for i in range(lo, hi+1):
                self._checked.add(i)
        else:
            # Plain click just toggles this one item (this dialog has no
            # separate "select single" mode — everything starts checked,
            # so a plain click is the natural way to uncheck/recheck one)
            if index in self._checked: self._checked.discard(index)
            else: self._checked.add(index)
            self._last_click_idx = index

        for i, t in self._thumbs.items():
            t.set_checked(i in self._checked)
        total = len(self._thumbs)
        self.status_lbl.setText(f"Selected {len(self._checked)} of {total} files")
        # Note: the preview pane navigates all LOADED images independent
        # of check state (per spec — arrows browse everything, not just
        # what's checked for export), so toggling a checkbox does not
        # need to touch the preview pane at all.

    def _export_all(self):
        self.export_mode = 'all'
        self.accept()

    def _export_selected(self):
        if not self._checked:
            QMessageBox.information(self, "Nothing Selected",
                "Select at least one image, or use Export all.")
            return
        self.export_mode = 'selected'
        self.accept()

    def get_export_paths(self) -> List[str]:
        """Ordered list of extracted file paths for the chosen export mode."""
        indices = (sorted(self._paths.keys()) if self.export_mode == 'all'
                   else sorted(self._checked))
        return [self._paths[i] for i in indices if self._paths.get(i)]

# ── Zip reader dialog (manga/comic-style page viewer) ─────────────────────────

class ZipReaderDialog(QDialog):
    """
    Full page viewer for images inside a zip. Matches the reader mockup:
    filename in the titlebar, the current page filling the view (scaled
    to fit, aspect preserved), a page counter, A1/A2/Off enhancement
    toggle, prev/next buttons, and Close.
    Navigable via on-screen buttons, Left/Right arrow keys, and clicking the
    left/right half of the image itself.

    A1 (mild) / A2 (strong) pre-process ALL pages in this zip session
    upfront (per spec) via PageEnhanceWorker, so once processing finishes,
    flipping between pages is instant. Off shows the original files with
    no processing delay. Enhancement choice here is independent of, and
    never affects, PDF export — this dialog never writes into the library
    or touches the zip's original files.
    """

    def __init__(self, page_paths: List[str], start_index: int = 0, parent=None):
        super().__init__(parent)
        self.page_paths = page_paths
        self.index = max(0, min(start_index, len(page_paths)-1)) if page_paths else 0
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        self.setMinimumSize(700, 550)
        self.resize(1000, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        self.setFocusPolicy(Qt.StrongFocus)
        self._maximize_on_show = True   # consumed once in showEvent — see below

        self._enhance_level: str = 'off'         # 'mild'|'strong'|'off'
        self._enhanced_paths: Dict[str, str] = {}  # original -> enhanced path
        self._enhance_worker: Optional[PageEnhanceWorker] = None
        self._enhance_dir = os.path.join(
            get_session_tmp_dir(), f"reader_enhance_{id(self)}")

        self._build_ui()
        self._show_page()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        self.title_lbl = QLabel("")
        self.title_lbl.setStyleSheet(
            f"background:{PANEL_BG}; color:{TEXT_PRI}; font-size:15px; "
            f"font-weight:bold; padding:10px; border-bottom:2px solid {BORDER};")
        outer.addWidget(self.title_lbl)

        # Image area — click left/right half to navigate, like most readers
        self.image_frame = QFrame()
        self.image_frame.setStyleSheet(f"background:{DARK_BG};")
        self.image_frame.setCursor(Qt.PointingHandCursor)
        img_layout = QVBoxLayout(self.image_frame)
        img_layout.setContentsMargins(20,20,20,20)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"background:{CARD_BG}; border:1px solid {BORDER};")
        img_layout.addWidget(self.image_label)
        self.image_frame.mousePressEvent = self._on_image_click
        outer.addWidget(self.image_frame, 1)

        bottom = QWidget()
        bottom.setStyleSheet(f"background:{PANEL_BG}; border-top:1px solid {BORDER};")
        brow = QHBoxLayout(bottom); brow.setContentsMargins(14,10,14,10)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        brow.addWidget(self.status_lbl)
        brow.addStretch()

        # A1 / A2 / Off — matches the reader mockup exactly
        self.enh_group = QButtonGroup(self)
        self.enh_buttons = {}
        for key, label, tip in [('mild','A1','Auto mild sharpen/contrast'),
                                ('strong','A2','Auto strong sharpen/contrast'),
                                ('off','Off','No enhancement')]:
            btn = QPushButton(label); btn.setCheckable(True)
            btn.setMinimumWidth(ws(44)); btn.setToolTip(tip)
            btn.clicked.connect(lambda _,k=key: self._set_enhance_level(k))
            self.enh_group.addButton(btn)
            self.enh_buttons[key] = btn
            brow.addWidget(btn)
        self.enh_buttons['off'].setChecked(True)

        brow.addSpacing(14)
        self.prev_btn = QPushButton("<"); self.prev_btn.setMinimumWidth(ws(40))
        self.prev_btn.clicked.connect(self.prev_page)
        brow.addWidget(self.prev_btn)
        self.next_btn = QPushButton(">"); self.next_btn.setMinimumWidth(ws(40))
        self.next_btn.clicked.connect(self.next_page)
        brow.addWidget(self.next_btn)
        brow.addSpacing(14)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept)
        brow.addWidget(close_btn)
        outer.addWidget(bottom)

    def _set_enhance_level(self, level: str):
        if level == self._enhance_level:
            return
        self._enhance_level = level
        if level == 'off':
            self._render_current()
            return
        self._start_enhance_preprocessing(level)

    def _start_enhance_preprocessing(self, level: str):
        """Pre-process ALL pages for this session at the chosen level.
        Runs in a background thread; a progress dialog blocks interaction
        until done, matching 'render all pages in auto (pre-processing)'."""
        sharpen_pct, contrast_f, noise_pct = READER_ENHANCE_PRESETS.get(level, (100, 1.0, 0))

        dlg = ProgressDialog(f"Enhancing pages ({self.enh_buttons[level].text()})…", self)
        worker = PageEnhanceWorker(self.page_paths, sharpen_pct, contrast_f,
                                   self._enhance_dir, noise_pct)
        self._enhance_worker = worker

        def on_prog(cur, tot, name):
            if dlg.cancelled: worker.cancel()
            dlg.update_progress(cur, tot, name)

        self._enhance_result = {}
        def on_done(mapping):
            self._enhance_result = mapping
            dlg.accept()

        worker.progress.connect(on_prog)
        worker.finished_sig.connect(on_done)
        worker.error_sig.connect(lambda m: log.error("page enhance: %s", m))
        worker.start(); dlg.exec_()

        if dlg.cancelled:
            # Reverting to Off — pre-processing was cancelled partway
            self.enh_buttons['off'].setChecked(True)
            self._enhance_level = 'off'
        else:
            self._enhanced_paths = self._enhance_result
        self._render_current()

    def _current_pixmap_source(self) -> Optional[QPixmap]:
        if not self.page_paths: return None
        path = self.page_paths[self.index]
        if self._enhance_level != 'off':
            path = self._enhanced_paths.get(path, path)
        pix = QPixmap(path)
        return pix if not pix.isNull() else None

    def _show_page(self):
        if not self.page_paths:
            self.title_lbl.setText("No images")
            self.status_lbl.setText("0/0 files")
            self.image_label.setText("No images found in this zip.")
            self.prev_btn.setEnabled(False); self.next_btn.setEnabled(False)
            return

        path = self.page_paths[self.index]
        fname = os.path.basename(path).split('_', 1)[-1]
        self.title_lbl.setText(fname)
        self.setWindowTitle(fname)
        self.status_lbl.setText(f"{self.index+1}/{len(self.page_paths)} files")
        self.prev_btn.setEnabled(self.index > 0)
        self.next_btn.setEnabled(self.index < len(self.page_paths)-1)
        self._render_current()

    def _render_current(self):
        pix = self._current_pixmap_source()
        if pix is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Could not load image.")
            return
        avail = self.image_frame.size()
        scaled = pix.scaled(avail.width()-40, avail.height()-40,
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current()   # rescale current page to the new size

    def _on_image_click(self, event):
        # Left half of the image = previous page, right half = next page
        if event.x() < self.image_frame.width() / 2:
            self.prev_page()
        else:
            self.next_page()

    def prev_page(self):
        if self.index > 0:
            self.index -= 1
            self._show_page()

    def next_page(self):
        if self.index < len(self.page_paths) - 1:
            self.index += 1
            self._show_page()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Up, Qt.Key_PageUp):
            self.prev_page()
        elif event.key() in (Qt.Key_Right, Qt.Key_Down, Qt.Key_PageDown, Qt.Key_Space):
            self.next_page()
        elif event.key() == Qt.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        # Maximizing in __init__ is unreliable — the window isn't realized
        # yet, so the OS/WM can ignore it. Doing it here, on the FIRST
        # showEvent only (flag consumed so re-showing after a modal child
        # dialog closes doesn't re-force maximize over a user's resize),
        # is the standard reliable pattern.
        super().showEvent(event)
        if getattr(self, '_maximize_on_show', False):
            self._maximize_on_show = False
            self.showMaximized()


# ── Standalone image viewer (library images, not inside a zip) ───────────────

class ImageViewerDialog(QDialog):
    """
    Full-window preview for standalone image files in the library — same
    look and navigation as ZipReaderDialog, but sourced from real file
    paths on disk (abs_path from the DB) rather than a temp-extracted zip.
    Next/Previous cycles through every image in the current folder view
    (per spec — same set as the gallery grid, filtered to images only).

    Reuses the same PageEnhanceWorker/enhance_image/ENHANCE_PRESETS as the
    zip reader for the A1/A2/Off toggle — pre-processes all images in this
    session upfront, same as the zip reader does for its pages.
    """

    def __init__(self, image_paths: List[str], start_index: int = 0, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.index = max(0, min(start_index, len(image_paths)-1)) if image_paths else 0
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        self.setMinimumSize(700, 550)
        self.resize(1000, 800)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        self.setFocusPolicy(Qt.StrongFocus)
        self._maximize_on_show = True   # consumed once in showEvent — see below

        self._enhance_level: str = 'off'          # 'mild'|'strong'|'off'
        self._enhanced_paths: Dict[str, str] = {}   # original -> enhanced path
        self._enhance_worker: Optional[PageEnhanceWorker] = None
        self._enhance_dir = os.path.join(
            get_session_tmp_dir(), f"image_enhance_{id(self)}")

        self._build_ui()
        self._show_page()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        self.title_lbl = QLabel("")
        self.title_lbl.setStyleSheet(
            f"background:{PANEL_BG}; color:{TEXT_PRI}; font-size:15px; "
            f"font-weight:bold; padding:10px; border-bottom:2px solid {BORDER};")
        outer.addWidget(self.title_lbl)

        # Image area — click left/right half to navigate, same as the reader
        self.image_frame = QFrame()
        self.image_frame.setStyleSheet(f"background:{DARK_BG};")
        self.image_frame.setCursor(Qt.PointingHandCursor)
        img_layout = QVBoxLayout(self.image_frame)
        img_layout.setContentsMargins(20,20,20,20)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"background:{CARD_BG}; border:1px solid {BORDER};")
        img_layout.addWidget(self.image_label)
        self.image_frame.mousePressEvent = self._on_image_click
        outer.addWidget(self.image_frame, 1)

        bottom = QWidget()
        bottom.setStyleSheet(f"background:{PANEL_BG}; border-top:1px solid {BORDER};")
        brow = QHBoxLayout(bottom); brow.setContentsMargins(14,10,14,10)
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        brow.addWidget(self.status_lbl)
        brow.addStretch()

        # A1 / A2 / Off — same enhancement toggle as the zip reader
        self.enh_group = QButtonGroup(self)
        self.enh_buttons = {}
        for key, label, tip in [('mild','A1','Auto mild sharpen/contrast'),
                                ('strong','A2','Auto strong sharpen/contrast'),
                                ('off','Off','No enhancement')]:
            btn = QPushButton(label); btn.setCheckable(True)
            btn.setMinimumWidth(ws(44)); btn.setToolTip(tip)
            btn.clicked.connect(lambda _,k=key: self._set_enhance_level(k))
            self.enh_group.addButton(btn)
            self.enh_buttons[key] = btn
            brow.addWidget(btn)
        self.enh_buttons['off'].setChecked(True)

        brow.addSpacing(14)
        self.prev_btn = QPushButton("<"); self.prev_btn.setMinimumWidth(ws(40))
        self.prev_btn.clicked.connect(self.prev_page)
        brow.addWidget(self.prev_btn)
        self.next_btn = QPushButton(">"); self.next_btn.setMinimumWidth(ws(40))
        self.next_btn.clicked.connect(self.next_page)
        brow.addWidget(self.next_btn)
        brow.addSpacing(14)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept)
        brow.addWidget(close_btn)
        outer.addWidget(bottom)

    def _set_enhance_level(self, level: str):
        if level == self._enhance_level:
            return
        self._enhance_level = level
        if level == 'off':
            self._render_current()
            return
        self._start_enhance_preprocessing(level)

    def _start_enhance_preprocessing(self, level: str):
        """Pre-process ALL images in this session upfront — same pattern
        as the zip reader's A1/A2 toggle."""
        sharpen_pct, contrast_f, noise_pct = READER_ENHANCE_PRESETS.get(level, (100, 1.0, 0))

        dlg = ProgressDialog(f"Enhancing images ({self.enh_buttons[level].text()})…", self)
        worker = PageEnhanceWorker(self.image_paths, sharpen_pct, contrast_f,
                                   self._enhance_dir, noise_pct)
        self._enhance_worker = worker

        def on_prog(cur, tot, name):
            if dlg.cancelled: worker.cancel()
            dlg.update_progress(cur, tot, name)

        self._enhance_result = {}
        def on_done(mapping):
            self._enhance_result = mapping
            dlg.accept()

        worker.progress.connect(on_prog)
        worker.finished_sig.connect(on_done)
        worker.error_sig.connect(lambda m: log.error("image enhance: %s", m))
        worker.start(); dlg.exec_()

        if dlg.cancelled:
            self.enh_buttons['off'].setChecked(True)
            self._enhance_level = 'off'
        else:
            self._enhanced_paths = self._enhance_result
        self._render_current()

    def _current_pixmap_source(self) -> Optional[QPixmap]:
        if not self.image_paths: return None
        path = self.image_paths[self.index]
        if self._enhance_level != 'off':
            path = self._enhanced_paths.get(path, path)
        pix = QPixmap(path)
        return pix if not pix.isNull() else None

    def _show_page(self):
        if not self.image_paths:
            self.title_lbl.setText("No images")
            self.status_lbl.setText("0/0 files")
            self.image_label.setText("No images found in this folder.")
            self.prev_btn.setEnabled(False); self.next_btn.setEnabled(False)
            return

        path = self.image_paths[self.index]
        fname = os.path.basename(path)
        self.title_lbl.setText(fname)
        self.setWindowTitle(fname)
        self.status_lbl.setText(f"{self.index+1}/{len(self.image_paths)} files")
        self.prev_btn.setEnabled(self.index > 0)
        self.next_btn.setEnabled(self.index < len(self.image_paths)-1)
        self._render_current()

    def _render_current(self):
        pix = self._current_pixmap_source()
        if pix is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Could not load image.")
            return
        avail = self.image_frame.size()
        scaled = pix.scaled(avail.width()-40, avail.height()-40,
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_current()

    def _on_image_click(self, event):
        if event.x() < self.image_frame.width() / 2:
            self.prev_page()
        else:
            self.next_page()

    def prev_page(self):
        if self.index > 0:
            self.index -= 1
            self._show_page()

    def next_page(self):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
            self._show_page()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Up, Qt.Key_PageUp):
            self.prev_page()
        elif event.key() in (Qt.Key_Right, Qt.Key_Down, Qt.Key_PageDown, Qt.Key_Space):
            self.next_page()
        elif event.key() == Qt.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_maximize_on_show', False):
            self._maximize_on_show = False
            self.showMaximized()


# ── Zip folder dialog (browse a zip's contents like a folder) ────────────────

class ZipFolderDialog(QDialog):
    """
    Opens a zip "like a folder" (per spec) — a grid of its images, streamed
    in progressively via ZipThumbLoader. Double-clicking any page opens
    ZipReaderDialog for full-page reading, starting at that page.

    Also shows a Tags panel (existing tags + search-and-add box) for the
    zip file itself, when db/file_id are provided — this is the only
    library-database interaction the dialog performs; browsing/extraction
    of the zip's own contents still never touches the library.
    """

    MIN_ZOOM_PCT, MAX_ZOOM_PCT, DEFAULT_ZOOM_PCT = 50, 300, 100

    def __init__(self, zip_path: str, zip_filename: str, parent=None,
                 db: Optional[LibraryDB] = None, file_id: Optional[int] = None):
        super().__init__(parent)
        self.zip_path = zip_path
        self.db = db
        self.file_id = file_id
        self.setWindowTitle(zip_filename)
        self.setMinimumSize(900, 650)
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint
                            | Qt.WindowMinimizeButtonHint)
        self.setFocusPolicy(Qt.StrongFocus)   # needed to receive wheel events reliably

        self._thumbs: Dict[int, ZipPageThumb] = {}
        self._paths:  Dict[int, str] = {}   # index -> extracted file path
        self._total: int = 0
        self._zoom_pct: int = self.DEFAULT_ZOOM_PCT

        self.extract_dir = os.path.join(
            get_session_tmp_dir(), f"browse_{os.path.basename(zip_path)}")

        self._build_ui(zip_filename)
        self._start_loading()
        if self.db and self.file_id is not None:
            self._refresh_tags()

    def _zoom_px(self, pct: int) -> int:
        return max(1, int(ZipPageThumb.DEFAULT_SIZE * pct / 100))

    def _build_ui(self, zip_filename: str):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0); outer.setSpacing(0)

        title_row = QWidget()
        title_row.setStyleSheet(f"background:{PANEL_BG}; border-bottom:2px solid {BORDER};")
        tl = QHBoxLayout(title_row); tl.setContentsMargins(14,8,14,8)
        title = QLabel(zip_filename)
        title.setStyleSheet(f"color:{TEXT_PRI}; font-size:15px; font-weight:bold; background:transparent;")
        tl.addWidget(title)
        tl.addStretch()
        zoom_lbl = QLabel("Size:")
        zoom_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        tl.addWidget(zoom_lbl)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(self.MIN_ZOOM_PCT, self.MAX_ZOOM_PCT)
        self.zoom_slider.setValue(self._zoom_pct)
        self.zoom_slider.setFixedWidth(140)
        self.zoom_slider.setToolTip(
            "Zoom out to see the whole zip at a glance, zoom in for detail.\n"
            "Ctrl+Scroll over the grid also zooms.")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        tl.addWidget(self.zoom_slider)
        self.zoom_pct_lbl = QLabel(f"{self._zoom_pct}%")
        self.zoom_pct_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent; min-width:36px;")
        tl.addWidget(self.zoom_pct_lbl)
        outer.addWidget(title_row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.viewport().installEventFilter(self)   # for Ctrl+Scroll zoom
        self.container = QWidget()
        self.grid = FlowGridLayout(self.container, h_spacing=10, v_spacing=10, margin=14)
        self.scroll.setWidget(self.container)
        outer.addWidget(self.scroll, 1)

        # Tags panel — only meaningful when this dialog knows which library
        # file it's showing (db + file_id supplied); otherwise stays hidden
        # since there'd be nothing to read or write tags against.
        self.tag_group = QGroupBox("Tags")
        tg = QVBoxLayout(self.tag_group)
        self.tag_flow_container = QWidget()
        self.tag_flow = FlowGridLayout(self.tag_flow_container, h_spacing=6, v_spacing=6, margin=4)
        tg.addWidget(self.tag_flow_container)

        add_row = QHBoxLayout()
        self.tag_search_edit = QLineEdit()
        self.tag_search_edit.setPlaceholderText("Type to search and add tags")
        add_row.addWidget(self.tag_search_edit, 1)
        add_btn = QPushButton("Add")
        add_btn.setObjectName("accent")
        add_btn.clicked.connect(self._on_add_tag_clicked)
        add_row.addWidget(add_btn)
        tg.addLayout(add_row)
        outer.addWidget(self.tag_group)

        if self.db and self.file_id is not None:
            self._tag_completer = TagFragmentCompleter(
                self.tag_search_edit, lambda: self.db.get_all_tag_categories(), self)
            self.tag_search_edit.returnPressed.connect(self._on_add_tag_clicked)
        else:
            self.tag_group.setVisible(False)

        bottom = QWidget()
        bottom.setStyleSheet(f"background:{PANEL_BG}; border-top:1px solid {BORDER};")
        brow = QHBoxLayout(bottom); brow.setContentsMargins(14,10,14,10)
        self.status_lbl = QLabel("Loading…")
        self.status_lbl.setStyleSheet(f"color:{TEXT_SEC}; background:transparent;")
        brow.addWidget(self.status_lbl)
        brow.addStretch()
        read_btn = QPushButton("Open reading"); read_btn.setObjectName("accent")
        read_btn.clicked.connect(self._open_reading_from_start)
        brow.addWidget(read_btn)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.accept)
        brow.addWidget(close_btn)
        outer.addWidget(bottom)

    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == event.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                delta = event.angleDelta().y()
                if delta != 0:
                    step = 10   # percent per notch — matches main gallery's granularity intent
                    new_pct = self._zoom_pct + (step if delta > 0 else -step)
                    new_pct = max(self.MIN_ZOOM_PCT, min(self.MAX_ZOOM_PCT, new_pct))
                    if new_pct != self._zoom_pct:
                        self.zoom_slider.setValue(new_pct)   # triggers _on_zoom_changed
                return True   # consume — don't also scroll the grid
        return super().eventFilter(obj, event)

    def _on_zoom_changed(self, value: int):
        self._zoom_pct = value
        self.zoom_pct_lbl.setText(f"{value}%")
        px = self._zoom_px(value)
        for t in self._thumbs.values():
            t.set_size(px)

    def _start_loading(self):
        self.loader = ZipThumbLoader(self.zip_path, self.extract_dir)
        self.loader.thumb_ready.connect(self._on_thumb_ready, Qt.QueuedConnection)
        self.loader.finished_sig.connect(self._on_load_finished, Qt.QueuedConnection)
        self.loader.error_sig.connect(self._on_load_error, Qt.QueuedConnection)
        self.loader.start()

    def _on_thumb_ready(self, index: int, path: str, qimg):
        fname = os.path.basename(path).split('_', 1)[-1] if path else f"page {index+1}"
        thumb = ZipPageThumb(index, fname, mode='browse', size=self._zoom_px(self._zoom_pct))
        thumb.double_clicked.connect(self._open_reader_at)
        thumb.set_thumbnail(qimg)
        self._thumbs[index] = thumb
        self._paths[index]  = path
        self.grid.addWidget(thumb)
        self.status_lbl.setText(f"Loaded {len(self._thumbs)} image(s)…")

    def _on_load_finished(self, total: int):
        self._total = total
        self.status_lbl.setText(
            f"Total {total} files" if total else "No images found in this zip.")

    def _on_load_error(self, msg: str):
        QMessageBox.warning(self, "Error", f"Could not read zip:\n{msg}")

    def _ordered_paths(self) -> List[str]:
        return [self._paths[i] for i in sorted(self._paths.keys()) if self._paths.get(i)]

    def _open_reader_at(self, index: int):
        """
        index is the zip-entry index (may have gaps if some pages failed
        to extract). Map it to a position in the FILTERED path list by
        counting how many successfully-extracted pages come at or before
        this index — not by position in the raw (unfiltered) key list,
        which would drift out of sync whenever any page failed.
        """
        paths = self._ordered_paths()
        if not paths: return
        start = sum(1 for k in sorted(self._paths.keys())
                    if k <= index and self._paths.get(k)) - 1
        start = max(0, min(start, len(paths)-1))
        ZipReaderDialog(paths, start_index=start, parent=self).exec_()

    def _open_reading_from_start(self):
        paths = self._ordered_paths()
        if not paths:
            QMessageBox.information(self, "No Images", "This zip has no images to read.")
            return
        ZipReaderDialog(paths, start_index=0, parent=self).exec_()

    # ── Tags panel ────────────────────────────────────────────────────────────

    def _refresh_tags(self):
        self.tag_flow.clear()
        for cat, sub in self.db.get_tags(self.file_id):
            box = _ZipTagBox(cat, sub)
            box.delete_requested.connect(self._on_delete_tag_requested)
            self.tag_flow.addWidget(box)

    def _on_delete_tag_requested(self, tag: Tuple[str, str]):
        cat, sub = tag
        reply = QMessageBox.question(
            self, "Delete Tag",
            f"Remove '{cat}: {sub}' from this item?",
            QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        current = [t for t in self.db.get_tags(self.file_id) if t != tag]
        self.db.set_tags(self.file_id, current)
        self._refresh_tags()

    def _on_add_tag_clicked(self):
        text = self.tag_search_edit.text().strip()
        if not text:
            return
        # Accept one or more comma-separated "category: subcategory" entries,
        # same syntax as everywhere else tags are typed in this app.
        added_any = False
        for part in [p.strip() for p in text.split(',') if p.strip()]:
            if ':' not in part:
                continue
            cat, sub = part.split(':', 1)
            cat, sub = cat.strip(), sub.strip()
            if not cat or not sub:
                continue
            self.db.bulk_assign_tag([self.file_id], cat, sub)
            added_any = True
        if added_any:
            self.tag_search_edit.clear()
            self._refresh_tags()
        else:
            QMessageBox.information(
                self, "Format", "Enter tags as 'category: subcategory', comma-separated.")

# ── Zip folder dialog: tag box (plain, right-click to delete) ────────────────

class _ZipTagBox(QFrame):
    """
    One 'Category: subcategory' box in ZipFolderDialog's Tags panel. Plain
    rectangle, no visible close icon — deletion is via right-click, which
    asks for confirmation, matching the per-item Tag Editor's spirit but
    inline rather than in a separate dialog.
    """
    delete_requested = pyqtSignal(tuple)   # (category, subcategory)

    def __init__(self, category: str, subcategory: str, parent=None):
        super().__init__(parent)
        self.tag = (category, subcategory)
        self.setStyleSheet(f"""
            _ZipTagBox {{
                background: {PANEL_BG};
                border: 1px solid {BORDER};
                border-radius: 5px;
            }}
            _ZipTagBox:hover {{ border-color: {ACCENT2}; }}
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lbl = QLabel(f"{category} : {subcategory}")
        lbl.setStyleSheet(f"color:{TEXT_PRI}; background:transparent;")
        lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Right-click to remove this tag")

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        del_act = menu.addAction("🗑  Delete Tag")
        action = menu.exec_(event.globalPos())
        if action == del_act:
            self.delete_requested.emit(self.tag)

# ── Progress dialog ───────────────────────────────────────────────────────────

class ProgressDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(ws(460), ws(130))
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        self.setWindowFlags(Qt.Dialog|Qt.WindowTitleHint|Qt.CustomizeWindowHint)
        layout=QVBoxLayout(self); layout.setContentsMargins(20,16,20,16)
        layout.setSpacing(10)
        self.status=QLabel("Starting…")
        self.status.setStyleSheet(f"color:{TEXT_SEC};font-size:{fs(12)}px;")
        layout.addWidget(self.status)
        self.bar=QProgressBar(); self.bar.setRange(0,100); layout.addWidget(self.bar)
        self.cancel_btn=QPushButton("Cancel"); self.cancel_btn.setMinimumWidth(ws(80))
        row=QHBoxLayout(); row.addStretch(); row.addWidget(self.cancel_btn)
        layout.addLayout(row); self.cancelled=False
        self.cancel_btn.clicked.connect(self._on_cancel)

    def update_progress(self, cur, tot, lbl):
        if tot>0: self.bar.setValue(int(cur*100/tot))
        fm=QFontMetrics(self.status.font())
        self.status.setText(fm.elidedText(f"[{cur}/{tot}] {lbl}",Qt.ElideMiddle,400))

    def _on_cancel(self):
        self.cancelled=True; self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling…")

# ── Tag autocomplete helper ───────────────────────────────────────────────────

class TagFragmentCompleter(QObject):
    """
    Attaches starts-with autocomplete to a QLineEdit that holds a
    comma-separated list of 'category: subcategory' fragments (the syntax
    used by the main search bar's tag box and elsewhere). Only the fragment
    currently being typed — the text since the last comma — is matched and
    completed; earlier, already-finished fragments are left alone.

    get_vocab_fn: callable returning the current List[Tuple[str,str]]
    vocabulary. Called fresh each time the popup is about to show, so it
    stays in sync with tags added/removed elsewhere (e.g. Tag Manager).
    """

    def __init__(self, line_edit: QLineEdit, get_vocab_fn, parent=None):
        super().__init__(parent)
        self.edit = line_edit
        self.get_vocab_fn = get_vocab_fn
        self.completer = QCompleter([], self.edit)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchStartsWith)
        self.completer.setCompletionMode(QCompleter.PopupCompletion)
        self.completer.setWidget(self.edit)
        self.completer.activated[str].connect(self._insert_completion)
        self.edit.textEdited.connect(self._on_text_edited)

    def _current_fragment_bounds(self) -> Tuple[int, int]:
        """Start/end index (in the full text) of the fragment the cursor
        is currently inside, delimited by commas."""
        text = self.edit.text()
        pos = self.edit.cursorPosition()
        start = text.rfind(',', 0, pos) + 1   # 0 if no comma found (rfind -> -1)
        end = text.find(',', pos)
        if end == -1:
            end = len(text)
        return start, end

    def _on_text_edited(self, _text: str):
        start, end = self._current_fragment_bounds()
        fragment = self.edit.text()[start:end].lstrip()
        if not fragment:
            self.completer.popup().hide()
            return

        vocab = self.get_vocab_fn()

        if ':' in fragment:
            # User has already typed (or is typing) "category:" — narrow
            # to that category's subcategories, same as before.
            cat_part, sub_part = fragment.split(':', 1)
            cat_part = cat_part.strip()
            sub_typed = sub_part.strip()
            candidates = [f"{c}: {s}" for c, s in vocab
                         if c.lower() == cat_part.lower()
                         and s.lower().startswith(sub_typed.lower())]
        else:
            # No colon yet — the user may be typing a SUBCATEGORY directly
            # (e.g. "car" meaning to reach "vehicle: car") rather than a
            # category name. Suggest every "category: subcategory" pair
            # whose subcategory starts with what's typed, across ALL
            # categories — so "car" surfaces "vehicle: car" even though
            # nothing about "vehicle" was typed. If the same subcategory
            # word exists under more than one category, all of them show
            # up as separate suggestions, letting the user pick the right
            # one rather than guessing which category was meant.
            typed = fragment.lower()
            candidates = [f"{c}: {s}" for c, s in vocab
                         if s.lower().startswith(typed)]
            # Also keep matching by category prefix (e.g. typing "veh"
            # toward "vehicle: car") so partial category names still work
            # even before a colon is typed — merge and dedupe, subcategory
            # matches first since that's the new, more useful behavior.
            cat_candidates = [f"{c}: {s}" for c, s in vocab
                              if c.lower().startswith(typed)
                              and f"{c}: {s}" not in candidates]
            candidates = candidates + cat_candidates

        if not candidates:
            self.completer.popup().hide()
            return

        self.completer.setModel(QStringListModel(candidates, self.completer))
        # IMPORTANT: candidates above are already manually filtered (by
        # subcategory-match, not necessarily by "starts with the full
        # fragment text"). QCompleter would otherwise re-apply its own
        # MatchStartsWith filter against the complete "category: sub"
        # string and hide everything, since e.g. "vehicle: car" does not
        # start with "car". Setting an empty prefix makes Qt's own filter
        # a no-op so it just shows the list we already built.
        self.completer.setCompletionPrefix("")
        self.completer.complete()

    def _insert_completion(self, completion: str):
        start, end = self._current_fragment_bounds()
        text = self.edit.text()
        prefix = text[:start]
        # Preserve a leading space after commas for readability, matching
        # how the field is normally typed ("animal: cats, color: white").
        if prefix and not prefix.endswith(' ') and prefix.rstrip().endswith(','):
            prefix = prefix.rstrip() + ' '
        suffix = text[end:]
        new_text = prefix + completion + suffix
        self.edit.setText(new_text)
        self.edit.setCursorPosition(len(prefix) + len(completion))

# ── Search bar ────────────────────────────────────────────────────────────────

class SearchBar(QWidget):
    search_changed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout=QHBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(8)
        self._base_widths = {}   # attr -> base px width, for live rescaling later
        for icon,tip,attr,w in [
            ("🔍","","name_edit",220),("🏷","","tag_edit",300)]:
            lbl=QLabel(icon); lbl.setStyleSheet(f"color:{TEXT_SEC};background:transparent;")
            layout.addWidget(lbl)
            ed=QLineEdit(); ed.setMinimumWidth(ws(w)); ed.textChanged.connect(self._emit)
            self._base_widths[attr] = w
            setattr(self,attr,ed); layout.addWidget(ed)
        self.name_edit.setPlaceholderText("Search by filename…")
        self.tag_edit.setPlaceholderText("Tags: animal: cats, -color: brown, show_only:3s+")
        self.tag_edit.setToolTip(
            "category: subcategory, comma-sep. Prefix - to exclude. AND logic.\n"
            "Ratings: show_only:5s (exactly 5 stars), show_only:2s+ (2+ stars),\n"
            "show_only:3h+ (3+ hearts), show_only:1h (exactly 1 heart)")
        self.rst_btn=QPushButton("✕ Reset"); self.rst_btn.setMinimumWidth(ws(90))
        self._base_widths['rst_btn'] = 90
        self.rst_btn.clicked.connect(self._reset); layout.addWidget(self.rst_btn)
        self._completer: Optional[TagFragmentCompleter] = None

    def refresh_text_scale(self):
        """Re-applies the current global UI_TEXT_SCALE to this bar's fixed-
        content widgets. SearchBar is built once at MainWindow startup and
        stays alive for the whole session, so — unlike dialogs that are
        freshly constructed (and pick up ws() at construction time) every
        time they're opened — it needs an explicit refresh hook for when
        the user changes text scale while the app is already running."""
        self.name_edit.setMinimumWidth(ws(self._base_widths['name_edit']))
        self.tag_edit.setMinimumWidth(ws(self._base_widths['tag_edit']))
        self.rst_btn.setMinimumWidth(ws(self._base_widths['rst_btn']))

    def enable_tag_autocomplete(self, get_vocab_fn):
        """Attach starts-with, per-fragment autocomplete to the tag box.
        get_vocab_fn is called fresh each time suggestions are shown, so
        it should return the current List[Tuple[str,str]] vocabulary
        (e.g. self.db.get_all_tag_categories)."""
        self._completer = TagFragmentCompleter(self.tag_edit, get_vocab_fn, self)

    def _emit(self):
        self.search_changed.emit(self.name_edit.text(), self.tag_edit.text())

    def _reset(self):
        self.name_edit.clear(); self.tag_edit.clear()

    @staticmethod
    def parse_tags(query: str):
        """
        Returns (inc_tags, exc_tags, rating_filters).
        rating_filters is a list of (kind, op, value):
          kind  = 'star' | 'heart'
          op    = 'eq' | 'gte'
          value = 1-5
        Syntax: show_only:5s  -> exactly 5 stars
                show_only:2s+ -> 2 or more stars
                show_only:3h+ -> 3 or more hearts
                show_only:1h  -> exactly 1 heart
        """
        inc=[]; exc=[]; ratings=[]
        for part in [p.strip() for p in query.split(',') if p.strip()]:
            low = part.lower()
            if low.startswith('show_only:'):
                spec = low[len('show_only:'):].strip()
                m = SearchBar._RATING_RE.match(spec)
                if m:
                    value = int(m.group(1))
                    kind  = 'star' if m.group(2) == 's' else 'heart'
                    op    = 'gte' if m.group(3) == '+' else 'eq'
                    ratings.append((kind, op, max(0,min(5,value))))
                continue
            neg=part.startswith('-')
            if neg: part=part[1:].strip()
            if ':' not in part: continue
            cat,sub=part.split(':',1)
            cat=cat.strip(); sub=sub.strip()
            if cat and sub:
                (exc if neg else inc).append((cat,sub))
        return inc,exc,ratings

    _RATING_RE = re.compile(r'^(\d)(s|h)(\+)?$')

# ── Breadcrumb bar ────────────────────────────────────────────────────────────

class BreadcrumbBar(QWidget):
    navigate = pyqtSignal(int)   # stack index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout=QHBoxLayout(self)
        self._layout.setContentsMargins(4,2,4,2)
        self._layout.setSpacing(2)
        self._path=[(None,"Library")]
        self._rebuild()

    def set_path(self, path):
        self._path=path; self._rebuild()

    def _rebuild(self):
        while self._layout.count():
            item=self._layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for i,(fid,name) in enumerate(self._path):
            if i>0:
                sep=QLabel("›"); sep.setStyleSheet(f"color:{TEXT_SEC};background:transparent;font-size:14px;")
                self._layout.addWidget(sep)
            is_last=(i==len(self._path)-1)
            btn=QPushButton(name); btn.setFlat(True)
            if is_last:
                btn.setStyleSheet(f"color:{TEXT_PRI};font-weight:bold;background:transparent;border:none;padding:2px 4px;")
                btn.setEnabled(False)
            else:
                btn.setStyleSheet(f"color:{ACCENT2};background:transparent;border:none;padding:2px 4px;text-decoration:underline;")
                _i=i; btn.clicked.connect(lambda _,idx=_i: self.navigate.emit(idx))
            self._layout.addWidget(btn)
        self._layout.addStretch()

# ── Welcome dialog ────────────────────────────────────────────────────────────

class WelcomeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("mlibrary — Welcome")
        self.setMinimumSize(ws(420), ws(260))
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE)); self.choice=None
        layout=QVBoxLayout(self); layout.setContentsMargins(40,36,40,36); layout.setSpacing(16)
        title=QLabel("mlibrary")
        title.setStyleSheet(f"color:{ACCENT};font-size:{fs(28)}px;font-weight:bold;background:transparent;")
        title.setAlignment(Qt.AlignCenter); layout.addWidget(title)
        sub=QLabel("Portable media library")
        sub.setStyleSheet(f"color:{TEXT_SEC};font-size:{fs(13)}px;background:transparent;")
        sub.setAlignment(Qt.AlignCenter); layout.addWidget(sub)
        layout.addSpacing(8)
        for txt,ch,obj in [("📁  Add Folder to New Library","add_folder","accent"),
                           ("📂  Open Existing Library","open","")]:
            btn=QPushButton(txt); btn.setMinimumHeight(ws(44))
            if obj: btn.setObjectName(obj)
            btn.clicked.connect(lambda _,c=ch: self._choose(c)); layout.addWidget(btn)

    def _choose(self,c): self.choice=c; self.accept()

# ── Import options dialog ─────────────────────────────────────────────────────

class ImportOptionsDialog(QDialog):
    def __init__(self, folder_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Options")
        self.setMinimumSize(ws(380), ws(200))
        self.setStyleSheet(build_app_style(UI_TEXT_SCALE)); self.recursive=False
        layout=QVBoxLayout(self); layout.setContentsMargins(24,20,24,20); layout.setSpacing(14)
        layout.addWidget(QLabel(f"Import: <b>{folder_name}</b>"))

        # Radio buttons in a QButtonGroup — Qt enforces mutual exclusivity
        # natively, so it's impossible for both to end up checked (which a
        # pair of independently-wired QCheckBoxes could not guarantee).
        self.btn_group = QButtonGroup(self)
        self.flat_rb=QRadioButton("Scan top-level folder only (faster)")
        self.flat_rb.setChecked(True)
        self.btn_group.addButton(self.flat_rb)
        layout.addWidget(self.flat_rb)

        self.rec_rb=QRadioButton("Scan recursively (include subfolders)")
        self.btn_group.addButton(self.rec_rb)
        self.rec_rb.toggled.connect(self._on_rec)
        layout.addWidget(self.rec_rb)

        # Belt-and-braces: force exclusivity explicitly rather than relying
        # solely on QButtonGroup defaults, and make it impossible to end up
        # with neither option selected.
        self.btn_group.setExclusive(True)

        self.warn=QLabel("⚠ Recursive scan may be slow for large folder trees.")
        self.warn.setStyleSheet(f"color:{WARN};font-size:12px;background:transparent;")
        self.warn.setVisible(False); layout.addWidget(self.warn)
        btns=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self._ok); btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_rec(self,checked): self.warn.setVisible(checked)

    def _ok(self):
        # Guard against an impossible-but-defensive blank state: if somehow
        # neither radio ends up checked, fall back to the safe default
        # (flat scan) instead of silently proceeding with recursive=False
        # by accident of an unset button.
        if not self.flat_rb.isChecked() and not self.rec_rb.isChecked():
            self.flat_rb.setChecked(True)
        self.recursive=self.rec_rb.isChecked()
        self.accept()

# ── Main window ───────────────────────────────────────────────────────────────

def path_display(path):
    parts=path.split(os.sep)
    return os.sep.join(['…']+parts[-2:]) if len(parts)>3 else path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mlibrary"); self.setMinimumSize(900,600); self.resize(1280,800)
        self.script_dir=os.path.dirname(os.path.abspath(__file__))
        self.lib_path=os.path.join(self.script_dir,LIBRARY_FILE)
        self.cfg_path=os.path.join(self.script_dir,CONFIG_FILE)
        self.config=AppConfig(self.cfg_path)
        self.db: Optional[LibraryDB]=None
        self._import_worker: Optional[ImportWorker]=None
        self._health_worker: Optional[HealthCheckWorker]=None

        # Restore persisted text scale before building any UI, so the
        # first paint already reflects it rather than starting at 100%
        # and jumping once the user's saved preference loads.
        global UI_TEXT_SCALE
        try:
            UI_TEXT_SCALE = max(1.0, min(2.0, float(self.config.get('text_scale','1.0'))))
        except (TypeError, ValueError):
            UI_TEXT_SCALE = 1.0
        self._apply_text_scale(UI_TEXT_SCALE, persist=False)   # already persisted; just apply

        self._build_ui()
        # Minimum width should fit the whole toolbar (Add Folder, Open
        # Library, both search boxes, Reset, Options, Size slider) without
        # anything getting clipped or overlapping — computed from the
        # toolbar's own sizeHint() rather than a guessed constant, so it
        # stays correct if the toolbar's contents or text scale change.
        tb_width = self.main_toolbar.sizeHint().width()
        self.setMinimumWidth(max(900, tb_width + 24))
        self._init_library()

    def _build_ui(self):
        tb=QToolBar("Main"); tb.setMovable(False); self.addToolBar(tb)
        self.main_toolbar = tb   # kept for sizeHint() lookups (e.g. minimum window width)
        for label,slot in [("➕ Add Folder",self._add_folder),
                            ("📂 Open Library",self._open_library)]:
            a=QAction(label,self); a.triggered.connect(slot); tb.addAction(a)
        tb.addSeparator()
        self.search_bar=SearchBar()
        self.search_bar.enable_tag_autocomplete(
            lambda: self.db.get_all_tag_categories() if self.db else [])
        self.search_bar.search_changed.connect(self._on_search); tb.addWidget(self.search_bar)
        tb.addSeparator()

        # ── Organize menu (group / sort / rescan / compact) ─────────────────────
        organize_btn=QPushButton("⚙ Options")
        organize_menu=QMenu(organize_btn)

        group_menu=organize_menu.addMenu("Group by")
        self._group_actions={}
        for label,key in [("None","none"),("Date Modified","date"),
                          ("File Type","type"),("File Size","size"),
                          ("Favorites","favorites")]:
            act=QAction(label,self,checkable=True)
            act.setChecked(key=='none')
            act.triggered.connect(lambda _,k=key: self._set_group_by(k))
            group_menu.addAction(act)
            self._group_actions[key]=act

        sort_menu=organize_menu.addMenu("Sort by")
        self._sort_actions={}
        for label,key in [("None (default)","none"),("Date Modified","date"),
                          ("File Type","type"),("File Size","size"),
                          ("Favorites","favorites")]:
            act=QAction(label,self,checkable=True)
            act.setChecked(key=='none')
            act.triggered.connect(lambda _,k=key: self._set_sort_by(k))
            sort_menu.addAction(act)
            self._sort_actions[key]=act

        sort_menu.addSeparator()
        self._sort_dir_actions={}
        for label,is_asc in [("↓ Descending (default)",False),("↑ Ascending",True)]:
            act=QAction(label,self,checkable=True)
            act.setChecked(is_asc==False)
            act.triggered.connect(lambda _,a=is_asc: self._set_sort_ascending(a))
            sort_menu.addAction(act)
            self._sort_dir_actions[is_asc]=act

        organize_menu.addSeparator()
        rescan_act=organize_menu.addAction("🔄  Rescan Library")
        rescan_act.triggered.connect(self._rescan_library)
        compact_act=organize_menu.addAction("🧹  Compact Database…")
        compact_act.triggered.connect(self._compact_library)

        organize_menu.addSeparator()
        tag_mgr_act=organize_menu.addAction("🏷  Tag Manager…")
        tag_mgr_act.triggered.connect(self._open_tag_manager)
        bulk_tag_act=organize_menu.addAction("📌  Bulk Assign Tags…")
        bulk_tag_act.triggered.connect(self._open_bulk_tag_assign)

        organize_menu.addSeparator()
        info_act=organize_menu.addAction("ℹ️  Library Information…")
        info_act.triggered.connect(self._open_library_info)

        organize_menu.addSeparator()
        text_size_menu=organize_menu.addMenu("🔠  Text Size")
        self._text_scale_actions={}
        _current_pct = int(round(UI_TEXT_SCALE*100))
        for pct in (100,125,150,175,200):
            act=QAction(f"{pct}%",self,checkable=True)
            act.setChecked(pct==_current_pct)
            act.triggered.connect(lambda _,p=pct: self._apply_text_scale(p/100.0))
            text_size_menu.addAction(act)
            self._text_scale_actions[pct]=act
        text_size_menu.addSeparator()
        custom_act=text_size_menu.addAction("Custom…")
        custom_act.triggered.connect(self._open_text_scale_dialog)

        organize_btn.setMenu(organize_menu)
        tb.addWidget(organize_btn)
        tb.addSeparator()

        self.size_label=QLabel("  Size: "); self.size_label.setStyleSheet(f"color:{TEXT_SEC};background:transparent;")
        tb.addWidget(self.size_label)
        self.scale_slider=QSlider(Qt.Horizontal)
        self.scale_slider.setRange(25,200)
        self.scale_slider.setValue(int(float(self.config.get('thumb_scale','1.0'))*100))
        self.scale_slider.setFixedWidth(150)
        self.scale_slider.setToolTip("Thumbnail zoom 0.25× – 2× (or Ctrl+Scroll in the gallery)")
        self.scale_slider.valueChanged.connect(self._on_scale)
        tb.addWidget(self.scale_slider)
        self.scale_lbl=QLabel("100%")
        self.scale_lbl.setStyleSheet(f"color:{TEXT_SEC};background:transparent;min-width:40px;")
        tb.addWidget(self.scale_lbl)

        central=QWidget(); self.setCentralWidget(central)
        ml=QVBoxLayout(central); ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)

        self.breadcrumb=BreadcrumbBar()
        self.breadcrumb.setStyleSheet(f"background:{PANEL_BG};border-bottom:1px solid {BORDER};")
        self.breadcrumb.navigate.connect(self._breadcrumb_nav); ml.addWidget(self.breadcrumb)

        self.gallery=GalleryCanvas(None, self.config)
        self.gallery.selection_changed.connect(self._on_sel)
        self.gallery.item_double_click.connect(self._open_file)
        self.gallery.item_right_click.connect(self._context_menu)
        self.gallery.navigate_folder.connect(self._on_folder_nav)
        self.gallery.zoom_changed.connect(self._on_gallery_zoom)
        self.gallery.set_text_scale(UI_TEXT_SCALE)   # apply persisted scale now that it exists
        ml.addWidget(self.gallery)

        self.status_label=QLabel("No library loaded")
        self.statusBar().addPermanentWidget(self.status_label,1)
        self.sel_label=QLabel("")
        self.statusBar().addPermanentWidget(self.sel_label)

    # ── Library lifecycle ─────────────────────────────────────────────────────

    def _init_library(self):
        if os.path.exists(self.lib_path): self._load_library(self.lib_path)
        else: self._show_welcome()

    def _show_welcome(self):
        dlg=WelcomeDialog(self)
        if dlg.exec_()==QDialog.Accepted:
            if dlg.choice=="add_folder": self._create_new_library(); self._add_folder()
            elif dlg.choice=="open": self._open_library()
        else:
            self.status_label.setText("No library. Use toolbar to open or add files.")

    def _create_new_library(self):
        if self.db: self.db.close()
        self.db=LibraryDB(self.lib_path)
        self.gallery.db=self.db; self.config.set('library_path',self.lib_path)

    def _load_library(self, path: str):
        log.info("Loading library: %s", path)
        if self.db: self.db.close()
        try:
            self.db=LibraryDB(path); self.lib_path=path
            self.gallery.db=self.db; self.config.set('library_path',path)
            log.info("Opened OK — %d items", self.db.file_count())
            self._run_health_check()
        except Exception as e:
            log.exception("_load_library failed")
            QMessageBox.critical(self,"Error",f"Could not open library:\n{e}")

    def _run_health_check(self):
        log.info("Health check: %s", self.lib_path)
        dlg=ProgressDialog("Checking Library…",self)
        worker=HealthCheckWorker(self.lib_path); self._health_worker=worker

        def on_prog(cur,tot,name):
            if dlg.cancelled: worker.cancel()
            dlg.update_progress(cur,tot,name)

        def on_done(issues):
            log.info("Health check done: %d issues", issues)
            try: dlg.accept()
            except RuntimeError: pass
            cnt=self.db.file_count()
            if issues>0:
                QMessageBox.information(self,"Library Check",
                    f"{issues} missing file(s) marked deleted.\nLibrary: {cnt} items.")
            self._refresh_gallery()
            self.status_label.setText(f"Library: {cnt} items  |  {path_display(self.lib_path)}")

        def on_err(msg):
            log.error("Health check error: %s", msg)
            try: dlg.accept()
            except RuntimeError: pass

        worker.progress.connect(on_prog)
        worker.finished_sig.connect(on_done)
        worker.error_sig.connect(on_err)
        dlg.cancel_btn.clicked.connect(worker.cancel)
        worker.start(); dlg.exec_()

    def _refresh_gallery(self, name="", tags=""):
        if not self.db: return
        par=self.gallery.current_parent_id
        if name or tags:
            inc,exc,ratings=SearchBar.parse_tags(tags)
            rows=self.db.search_files(name,inc,exc,par,ratings)
        else:
            rows=self.db.get_files(parent_id=par)
        self.gallery.load_view(rows)
        self.breadcrumb.set_path(self.gallery.breadcrumb_path())
        cnt=self.db.file_count()
        self.status_label.setText(f"Library: {cnt} items  |  Showing: {len(rows)}")

    # ── Toolbar actions ───────────────────────────────────────────────────────

    def _add_folder(self):
        if not self.db: self._create_new_library()
        folder=QFileDialog.getExistingDirectory(self,"Select Folder","")
        if not folder: return
        sd=os.path.splitdrive(self.script_dir)[0].lower()
        fd=os.path.splitdrive(folder)[0].lower()
        if sd and fd and sd!=fd:
            # No longer blocks the import — cross-drive/cross-folder sources
            # are fully supported (files are keyed by absolute path when a
            # relative path can't be computed). This is just a heads-up:
            # moving the library+script to another machine won't carry
            # cross-drive files along automatically the way same-drive
            # relative paths would.
            log.info("Cross-drive import: library on '%s', folder on '%s'",
                     sd.upper(), fd.upper())
        opts=ImportOptionsDialog(os.path.basename(folder),self)
        if opts.exec_()!=QDialog.Accepted: return
        self.db.add_source(folder, opts.recursive)   # track for future rescans
        self._run_import([folder], opts.recursive)

    def _add_files(self):
        if not self.db: self._create_new_library()
        exts=" ".join(f"*{e}" for e in sorted(ALL_EXTS))
        files,_=QFileDialog.getOpenFileNames(self,"Select Files","",
            f"Media Files ({exts});;All Files (*.*)")
        if files: self._run_import(files, recursive=False)

    def _run_import(self, paths, recursive):
        dlg=ProgressDialog("Importing Files…",self)
        worker=ImportWorker(self.db,paths,self.script_dir,recursive,
                            self.gallery.current_parent_id)
        # Filled in by on_done; read AFTER dlg.exec_() returns below.
        self._import_result = (0, 0)

        def on_prog(cur,tot,name):
            if dlg.cancelled: worker.cancel()
            dlg.update_progress(cur,tot,name)

        def on_done(added,skipped):
            self._import_result = (added, skipped)
            dlg.accept()   # unblocks exec_() below — worker has already
                            # committed every DB write by the time this fires

        worker.progress.connect(on_prog)
        worker.finished_sig.connect(on_done)
        worker.error_sig.connect(lambda m: log.error("import: %s",m))
        worker.start(); self._import_worker=worker; dlg.exec_()

        # exec_() has now returned — the worker thread has fully finished
        # and every add_file/add_thumbnail commit is on disk. Refreshing
        # here (rather than inside on_done) guarantees the gallery query
        # sees every imported row, including ones from recursive subfolders.
        added, skipped = self._import_result
        self._refresh_gallery()
        QMessageBox.information(self,"Import Complete",
            f"Added: {added}  |  Skipped (already in library): {skipped}")

    def _open_library(self):
        path,_=QFileDialog.getOpenFileName(self,"Open Library",self.script_dir,
            "Library Files (*.lib);;All Files (*.*)")
        if path: self._load_library(path); self._refresh_gallery()

    # ── Scale ─────────────────────────────────────────────────────────────────

    def _on_scale(self, val):
        self.scale_lbl.setText(f"{val}%")
        self.gallery.set_scale(val/100.0)

    def _on_gallery_zoom(self, val_pct: int):
        """Ctrl+scroll in the gallery changed zoom — keep slider in sync
        without re-triggering set_scale a second time."""
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(val_pct)
        self.scale_slider.blockSignals(False)
        self.scale_lbl.setText(f"{val_pct}%")

    # ── Text size (app-wide UI scale, 100%-200%) ──────────────────────────────

    def _apply_text_scale(self, scale: float, persist: bool = True):
        """
        Central handler for the text-size setting. Updates the global
        UI_TEXT_SCALE, re-applies the app-wide stylesheet (covers every
        ordinary Qt widget/dialog automatically via QApplication-level
        cascade), pushes the new scale into GalleryCanvas (which paints
        its own text and needs its own hook), and persists the choice.
        """
        global UI_TEXT_SCALE
        scale = max(1.0, min(2.0, scale))
        UI_TEXT_SCALE = scale

        app = QApplication.instance()
        style = build_app_style(scale)
        if app is not None:
            app.setStyleSheet(style)     # cascades to every top-level window
        self.setStyleSheet(style)        # this window's own explicit sheet too

        if hasattr(self, 'gallery'):
            self.gallery.set_text_scale(scale)

        if hasattr(self, 'search_bar'):
            self.search_bar.refresh_text_scale()   # long-lived widget — needs an explicit poke

        if hasattr(self, 'text_scale_lbl'):
            self.text_scale_lbl.setText(f"{int(round(scale*100))}%")
        if hasattr(self, '_text_scale_actions'):
            pct = int(round(scale*100))
            for p, act in self._text_scale_actions.items():
                act.setChecked(p == pct)

        if persist:
            self.config.set('text_scale', str(scale))

    def _open_text_scale_dialog(self):
        """Prompt for a custom text scale (100-200%) via a small slider
        dialog, for values not covered by the preset menu entries."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Text Size")
        dlg.setMinimumSize(ws(360), ws(140))
        dlg.setStyleSheet(build_app_style(UI_TEXT_SCALE))
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.addWidget(QLabel("Text size:"))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(100, 200)
        slider.setValue(int(round(UI_TEXT_SCALE*100)))
        row.addWidget(slider)
        pct_lbl = QLabel(f"{slider.value()}%")
        pct_lbl.setMinimumWidth(ws(44))
        row.addWidget(pct_lbl)
        layout.addLayout(row)

        def on_slider(v):
            pct_lbl.setText(f"{v}%")
            self._apply_text_scale(v/100.0)
            # The dialog's own size and the label's minimum width should
            # keep pace with the scale being set live, same as every
            # other dialog does at the moment it's (re)constructed.
            pct_lbl.setMinimumWidth(ws(44))
            dlg.setMinimumSize(ws(360), ws(140))
        slider.valueChanged.connect(on_slider)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.accept); btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        dlg.exec_()

    # ── Organize: group / sort ──────────────────────────────────────────────────

    def _set_group_by(self, key: str):
        for k,act in self._group_actions.items(): act.setChecked(k==key)
        self.gallery._group_by = key
        self.gallery._collapsed_groups.clear()
        self.gallery._rebuild_layout()
        self.gallery._update_scrollbar()
        self.gallery.viewport().update()

    def _set_sort_by(self, key: str):
        for k,act in self._sort_actions.items(): act.setChecked(k==key)
        self.gallery._sort_by = key
        self.gallery._rebuild_layout()
        self.gallery._update_scrollbar()
        self.gallery.viewport().update()

    def _set_sort_ascending(self, ascending: bool):
        for is_asc,act in self._sort_dir_actions.items(): act.setChecked(is_asc==ascending)
        self.gallery._sort_ascending = ascending
        self.gallery._rebuild_layout()
        self.gallery._update_scrollbar()
        self.gallery.viewport().update()

    # ── Organize: rescan / compact ──────────────────────────────────────────────

    def _rescan_library(self):
        if not self.db:
            QMessageBox.information(self,"No Library","Open or create a library first.")
            return
        sources = self.db.get_sources()
        if not sources:
            QMessageBox.information(self,"Nothing to Rescan",
                "This library has no tracked source folders yet.\n"
                "Use Add Folder first.")
            return

        dlg=ProgressDialog("Rescanning Library…",self)
        worker=RescanWorker(self.lib_path, self.script_dir)

        def on_prog(cur,tot,name):
            if dlg.cancelled: worker.cancel()
            dlg.update_progress(cur,tot,name)

        self._rescan_result=(0,0,0,0,0)
        def on_done(added,restored,removed,updated,relinked):
            self._rescan_result=(added,restored,removed,updated,relinked)
            dlg.accept()

        worker.progress.connect(on_prog)
        worker.finished_sig.connect(on_done)
        worker.error_sig.connect(lambda m: log.error("rescan: %s",m))
        worker.start(); dlg.exec_()

        added,restored,removed,updated,relinked = self._rescan_result
        self._refresh_gallery()
        msg = (f"Added: {added}\nRestored: {restored}\n"
               f"Marked deleted: {removed}\nUpdated (changed): {updated}")
        if relinked:
            msg += f"\nRelinked (renamed/moved, tags kept): {relinked}"
        QMessageBox.information(self,"Rescan Complete",msg)

    def _compact_library(self):
        if not self.db:
            QMessageBox.information(self,"No Library","Open or create a library first.")
            return
        reply=QMessageBox.question(self,"Compact Database",
            "This permanently removes files marked as deleted (they will no "
            "longer be restorable by rescan) and reclaims disk space.\n\n"
            "This cannot be undone. Continue?",
            QMessageBox.Yes|QMessageBox.No)
        if reply!=QMessageBox.Yes: return

        # Close the main connection's WAL state isn't strictly required for
        # VACUUM, but keeping it simple: run compact then reopen is unnecessary
        # since CompactWorker uses its own connection to the same file.
        dlg=ProgressDialog("Compacting…",self)
        dlg.bar.setRange(0,0)   # indeterminate — compact has no per-item progress
        worker=CompactWorker(self.lib_path)

        self._compact_result=(0,0)
        def on_done(purged,reclaimed):
            self._compact_result=(purged,reclaimed)
            dlg.accept()

        worker.finished_sig.connect(on_done)
        worker.error_sig.connect(lambda m: log.error("compact: %s",m))
        worker.start(); dlg.exec_()

        purged,reclaimed = self._compact_result
        self._refresh_gallery()
        mb = reclaimed/1048576
        QMessageBox.information(self,"Compact Complete",
            f"Permanently removed {purged} deleted file(s).\n"
            f"Reclaimed approximately {mb:.1f} MB.")

    # ── Tag management ───────────────────────────────────────────────────────

    def _open_tag_manager(self):
        if not self.db:
            QMessageBox.information(self,"No Library","Open or create a library first.")
            return
        TagManagerDialog(self.db, self).exec_()
        # Tag renames/merges/deletes can change what's visible under the
        # current tag search filter, so refresh in case it's active.
        self._refresh_gallery(self.search_bar.name_edit.text(),
                              self.search_bar.tag_edit.text())

    def _open_bulk_tag_assign(self):
        if not self.db:
            QMessageBox.information(self,"No Library","Open or create a library first.")
            return
        if not self.db.get_all_tag_categories():
            QMessageBox.information(self,"No Tags Yet",
                "Create at least one tag in Tag Manager first.")
            return
        BulkTagAssignDialog(self.db, parent=self).exec_()
        self._refresh_gallery(self.search_bar.name_edit.text(),
                              self.search_bar.tag_edit.text())

    def _open_library_info(self):
        if not self.db:
            QMessageBox.information(self,"No Library","Open or create a library first.")
            return
        LibraryInfoDialog(self.db, self.lib_path, self).exec_()

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search(self, name, tags): self._refresh_gallery(name, tags)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _on_folder_nav(self, fid, name):
        self.breadcrumb.set_path(self.gallery.breadcrumb_path())
        cnt=len(self.gallery._rows)
        self.status_label.setText(f"Folder: {name}  |  {cnt} items")

    def _breadcrumb_nav(self, stack_idx: int):
        self.gallery.jump_to_stack_index(stack_idx)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_sel(self, ids):
        self.sel_label.setText(f"  Selected: {len(ids)}" if ids else "")

    # ── Context menu ──────────────────────────────────────────────────────────

    def _context_menu(self, fid: int):
        sel=self.gallery.get_selected_ids() or [fid]
        rows=[self.db.get_file_by_id(i) for i in sel]
        zip_ids=[r['id'] for r in rows if r and r['file_type']=='zip']

        menu=QMenu(self)
        ca=menu.addAction("📋  Copy to Folder…"); menu.addSeparator()
        ta=menu.addAction("🏷  Edit Tags…")
        da=menu.addAction("ℹ  File Details"); da.setEnabled(len(sel)==1)
        pa=None; oe=None
        if zip_ids:
            menu.addSeparator()
            oe=menu.addAction("📂  Open in File Explorer")
            oe.setEnabled(len(zip_ids)==1)
            pa=menu.addAction("📕  Export as PDF…")
        menu.addSeparator()
        rt=menu.addAction("♥★  Reset Ratings")
        menu.addSeparator()
        ra=menu.addAction("🗑  Remove from Library")
        action=menu.exec_(QCursor.pos())
        if action==ca: self._copy_to(sel)
        elif action==ta: TagEditorDialog(self.db,sel,self).exec_()
        elif action==da: FileDetailsDialog(self.db,sel[0],self).exec_()
        elif oe is not None and action==oe: self._open_in_file_explorer(zip_ids[0])
        elif pa is not None and action==pa: self._export_zips_as_pdf(zip_ids)
        elif action==rt: self._reset_ratings(sel)
        elif action==ra: self._remove(sel)

    def _reset_ratings(self, ids):
        for fid in ids: self.db.reset_ratings(fid)
        self.gallery.refresh_rows_from_db(ids)

    # ── Export zip(s) as PDF ─────────────────────────────────────────────────

    def _export_zips_as_pdf(self, zip_ids: List[int]):
        rows = [self.db.get_file_by_id(i) for i in zip_ids]
        rows = [r for r in rows if r and r['abs_path'] and os.path.exists(r['abs_path'])]
        if not rows:
            QMessageBox.warning(self,"No Files",
                "None of the selected zip files could be found on disk.")
            return

        if len(rows) == 1:
            self._export_single_zip_with_preview(rows[0])
        else:
            self._export_multiple_zips_direct(rows)

    def _export_single_zip_with_preview(self, row):
        dlg = ZipExportPreviewDialog(row['abs_path'], row['filename'], self)
        if dlg.exec_() != QDialog.Accepted or dlg.export_mode is None:
            return
        paths = dlg.get_export_paths()
        if not paths:
            QMessageBox.information(self,"Nothing to Export","No images were selected.")
            return
        sharpen_pct, contrast_f, noise_pct = dlg.get_enhance_params()

        out_dir = QFileDialog.getExistingDirectory(self,"Save PDF To…","")
        if not out_dir:
            return

        pdf_name = Path(row['filename']).stem + ".pdf"
        job = {'pdf_name': pdf_name, 'file_paths': paths}
        self._run_pdf_export([job], out_dir, sharpen_percent=sharpen_pct,
                             contrast_factor=contrast_f, noise_percent=noise_pct)

    def _export_multiple_zips_direct(self, rows):
        # No preview shown for multi-zip export (per spec), so no
        # enhancement controls either — exports at original quality.
        out_dir = QFileDialog.getExistingDirectory(
            self,"Save PDFs To… (one PDF per zip)","")
        if not out_dir:
            return
        jobs = [{'pdf_name': Path(r['filename']).stem + ".pdf",
                 'zip_path': r['abs_path']} for r in rows]
        self._run_pdf_export(jobs, out_dir)

    def _run_pdf_export(self, jobs: List[dict], out_dir: str,
                        sharpen_percent: float = 100, contrast_factor: float = 1.0,
                        noise_percent: float = 0):
        dlg=ProgressDialog("Exporting PDF…",self)
        worker=PdfExportWorker(jobs, out_dir, sharpen_percent, contrast_factor, noise_percent)

        def on_prog(cur,tot,label):
            if dlg.cancelled: worker.cancel()
            dlg.update_progress(cur,tot,label)

        self._pdf_export_result=(0,0)
        def on_done(created,failed):
            self._pdf_export_result=(created,failed)
            dlg.accept()

        worker.progress.connect(on_prog)
        worker.finished_sig.connect(on_done)
        worker.error_sig.connect(lambda m: log.error("pdf export: %s",m))
        worker.start(); dlg.exec_()

        created,failed = self._pdf_export_result
        msg = f"Created {created} PDF file(s) in:\n{out_dir}"
        if failed:
            msg += f"\n\n{failed} file(s) failed — see log for details."
        QMessageBox.information(self,"Export Complete",msg)

    def _copy_to(self, ids):
        dest=QFileDialog.getExistingDirectory(self,"Copy Files to Folder","")
        if not dest: return
        errors=[]; copied=0
        for fid in ids:
            row=self.db.get_file_by_id(fid)
            if not row or not row['abs_path']: errors.append(f"ID {fid}: no path"); continue
            src=row['abs_path']
            if not os.path.exists(src):
                errors.append(f"{row['filename']}: not found"); self.db.mark_deleted(fid); continue
            try: shutil.copy2(src,os.path.join(dest,row['filename'])); copied+=1
            except Exception as e: errors.append(f"{row['filename']}: {e}")
        msg=f"Copied {copied} file(s)."
        if errors: msg+=f"\n\nErrors:\n"+"\n".join(errors[:10])
        QMessageBox.information(self,"Copy Complete",msg)

    def _remove(self, ids):
        if QMessageBox.question(self,"Remove Files",
            f"Remove {len(ids)} file(s) from the library?\n(Originals are NOT deleted.)",
            QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        for fid in ids: self.db.mark_deleted(fid)
        self._refresh_gallery()

    # ── Open file ─────────────────────────────────────────────────────────────

    def _open_file(self, fid: int):
        row=self.db.get_file_by_id(fid)
        if not row or not row['abs_path']: return
        path=row['abs_path']
        if not os.path.exists(path):
            QMessageBox.warning(self,"File Not Found",
                f"Not found:\n{path}\nIt will be marked deleted.")
            self.db.mark_deleted(fid); self._refresh_gallery(); return

        if row['file_type'] == 'zip':
            # Double-clicking a zip opens it "like a folder" for reading —
            # file-explorer access moved to the right-click context menu.
            ZipFolderDialog(path, row['filename'], self, db=self.db, file_id=fid).exec_()
            return

        if row['file_type'] == 'image':
            # Standalone image: open the viewer with Next/Previous cycling
            # through every image in the CURRENT folder view (per spec) —
            # self.gallery._rows already reflects whatever folder/search/
            # sort/group state is active, so this matches what's on screen.
            #
            # Filter to (id, abs_path) pairs with a real path FIRST, then
            # find the clicked file's position within that same filtered
            # list — computing the index from the unfiltered row list and
            # applying it to the filtered path list would misalign them
            # if any image row happened to have no abs_path.
            image_items = [(r['id'], r['abs_path']) for r in self.gallery._rows
                          if r['file_type'] == 'image' and r['abs_path']]
            image_paths = [p for _, p in image_items]
            try:
                start = [i for i, _ in image_items].index(fid)
            except ValueError:
                start = 0   # fid not in current view for some reason — fall back safely
            ImageViewerDialog(image_paths, start_index=start, parent=self).exec_()
            return

        try: os.startfile(path)
        except AttributeError: subprocess.Popen(['xdg-open',path])
        except Exception as e: QMessageBox.warning(self,"Cannot Open",str(e))

    def _open_in_file_explorer(self, fid: int):
        """Right-click action for zip files: open with the OS default
        application (file explorer / archive manager) — the behavior
        double-click used to have before zips became browsable in-app."""
        row=self.db.get_file_by_id(fid)
        if not row or not row['abs_path']: return
        path=row['abs_path']
        if not os.path.exists(path):
            QMessageBox.warning(self,"File Not Found",
                f"Not found:\n{path}\nIt will be marked deleted.")
            self.db.mark_deleted(fid); self._refresh_gallery(); return
        try: os.startfile(path)
        except AttributeError: subprocess.Popen(['xdg-open',path])
        except Exception as e: QMessageBox.warning(self,"Cannot Open",str(e))

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.gallery._cancel_hover_loaders()
        self.gallery._cancel_fetchers()
        if self.db: self.db.close()
        self.config.save(); super().closeEvent(event)

# ── Entry point ───────────────────────────────────────────────────────────────

def _make_app_icon() -> QIcon:
    """Generate a programmatic icon — no external file needed."""
    sizes = [16, 32, 48, 64, 128, 256]
    icon  = QIcon()
    for sz in sizes:
        img = QImage(sz, sz, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        # Background circle
        p.setBrush(QBrush(QColor(ACCENT)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, sz, sz)
        # "mL" text
        p.setPen(QColor("white"))
        f = QFont("Segoe UI", max(int(sz * 0.32), 6), QFont.Bold)
        p.setFont(f)
        p.drawText(QRect(0, 0, sz, sz), Qt.AlignCenter, "mL")
        p.end()
        icon.addPixmap(QPixmap.fromImage(img))
    return icon

def main():
    # Windows: tell taskbar this is its own app, not python.exe
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("mlibrary.app.1")
    except Exception:
        pass   # not Windows — silently skip

    app=QApplication(sys.argv)
    app.setApplicationName("mlibrary"); app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    pal=QPalette()
    for role,col in [(QPalette.Window,DARK_BG),(QPalette.WindowText,TEXT_PRI),
                     (QPalette.Base,PANEL_BG),(QPalette.Text,TEXT_PRI),
                     (QPalette.Button,PANEL_BG),(QPalette.ButtonText,TEXT_PRI),
                     (QPalette.Highlight,ACCENT),(QPalette.HighlightedText,"white"),
                     (QPalette.ToolTipBase,PANEL_BG),(QPalette.ToolTipText,TEXT_PRI)]:
        pal.setColor(role,QColor(col))
    app.setPalette(pal)
    app_icon = _make_app_icon()
    app.setWindowIcon(app_icon)
    win=MainWindow(); win.setWindowIcon(app_icon); win.show(); sys.exit(app.exec_())

if __name__=="__main__":
    main()
