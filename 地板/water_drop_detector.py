"""
水滴辨識系統 — 統一 GUI 介面
================================
用法：
  python water_drop_detector.py [圖片資料夾]

操作方式：
  滑鼠拖曳       : 框選 ROI（先點「框選」按鈕或按 D）
  滾輪           : 縮放影像
  右鍵拖曳       : 平移（縮放後使用）
  雙擊左鍵       : 重設縮放
  SPACE          : 播放 / 暫停
  D              : 進入框選模式
  A              : 追加 ROI
  C              : 清除所有 ROI
  M              : 切換偵測方法
  +/-            : 調整 FPS
  ESC            : 離開
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from collections import deque
import sys
import argparse
import csv
import threading

# ─────────── 預設參數 ───────────
DEFAULT_FPS        = 5
BINARIZE_THRESH    = 200
TOPHAT_SIGMA       = 1.5
TOPHAT_KERNEL_SIZE = 61
ROW_THRESH_RATIO   = 1.3
ROI_PADDING        = 30
MIN_BLOB_AREA      = 5
MAX_BLOB_AREA      = 3000
HIST_WINDOW        = 60
HIST_CHANGE_THRESH = 0.25
SUPPORTED_EXT      = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

ROI_COLORS_BGR = [
    (0, 220, 80),
    (255, 165, 0),
    (0, 200, 255),
    (220, 0, 220),
    (0, 255, 255),
]

C_BG    = '#0f172a'
C_PANEL = '#1e293b'
C_CARD  = '#0d1117'
C_TEXT  = '#e2e8f0'
C_MUTED = '#64748b'
C_ACCENT= '#38bdf8'
C_GREEN = '#22c55e'
C_RED   = '#ef4444'
C_YELLOW= '#f59e0b'
C_PURPLE= '#6366f1'


# ─────────── 影像處理函式 ───────────

def read_image_unicode(path: Path):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[讀取失敗] {path.name}: {e}")
        return None


def collect_images(folder: Path):
    return sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXT],
        key=lambda p: p.name
    )


def auto_roi_rect(gray: np.ndarray):
    h, w = gray.shape
    row_mean = gray.mean(axis=1)
    thresh = row_mean.mean() * ROW_THRESH_RATIO
    bright = np.where(row_mean > thresh)[0]
    if len(bright) == 0:
        return (0, h // 3, w, 2 * h // 3)
    y1 = max(int(bright.min()) - ROI_PADDING, 0)
    y2 = min(int(bright.max()) + ROI_PADDING, h)
    return (0, y1, w, y2)


def detect_method_b(roi_gray: np.ndarray, sigma: float = TOPHAT_SIGMA):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE))
    tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, k)
    th = max(float(tophat.mean() + sigma * tophat.std()), 8.0)
    _, binary = cv2.threshold(tophat, int(th), 255, cv2.THRESH_BINARY)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ke, iterations=1)
    return binary, tophat


def detect_method_a(roi_gray: np.ndarray, thresh: int):
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    return binary


def find_blobs(binary: np.ndarray):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            blobs.append({'cx': centroids[i][0], 'cy': centroids[i][1], 'area': area})
    return blobs


def compute_hist(gray: np.ndarray):
    h = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    s = h.sum()
    return h / s if s > 0 else h


def chi2_dist(h1, h2):
    eps = 1e-10
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)) / 2)


def clamp_rect(x1, y1, x2, y2, W, H):
    x1, x2 = sorted([max(0, min(x1, W - 1)), max(0, min(x2, W - 1))])
    y1, y2 = sorted([max(0, min(y1, H - 1)), max(0, min(y2, H - 1))])
    return x1, y1, x2, y2


def draw_overlay(frame, rois, all_blobs, is_wet, paused, draw_mode):
    vis = frame.copy()
    h, w = vis.shape[:2]

    for idx, (rect, blobs) in enumerate(zip(rois, all_blobs)):
        x1, y1, x2, y2 = rect
        color = ROI_COLORS_BGR[idx % len(ROI_COLORS_BGR)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"ROI {idx+1}: {len(blobs)} drops"
        cv2.putText(vis, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        for b in blobs:
            cx = int(b['cx']) + x1
            cy = int(b['cy']) + y1
            r  = max(3, int(np.sqrt(b['area'] / np.pi)))
            cv2.circle(vis, (cx, cy), r, (0, 200, 255), 1)
            cv2.circle(vis, (cx, cy), 2, (0, 200, 255), -1)

    status_text = "WET" if is_wet else "DRY"
    sc = (0, 60, 255) if is_wet else (60, 220, 60)
    cv2.putText(vis, status_text, (w - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(vis, status_text, (w - 90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, sc, 2, cv2.LINE_AA)

    if draw_mode:
        msg = "[ Drag to select ROI ]"
        cv2.putText(vis, msg, (w // 2 - 100, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, msg, (w // 2 - 100, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 0), 1, cv2.LINE_AA)
    if paused:
        cv2.putText(vis, "PAUSED", (w // 2 - 55, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, "PAUSED", (w // 2 - 55, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2, cv2.LINE_AA)
    return vis


# ─────────── 輔助 UI 元件 ───────────

def flat_btn(parent, text, cmd, bg=C_ACCENT, fg='white',
             font_size=9, padx=10, pady=4, **kw):
    return tk.Button(
        parent, text=text, command=cmd, bg=bg, fg=fg,
        activebackground=bg, activeforeground=fg,
        relief='flat', bd=0, cursor='hand2',
        font=('微軟正黑體', font_size), padx=padx, pady=pady, **kw
    )


def section_label(parent, text):
    f = tk.Frame(parent, bg=C_PANEL)
    f.pack(fill='x', padx=10, pady=(8, 2))
    tk.Label(f, text=text, bg=C_PANEL, fg=C_ACCENT,
             font=('微軟正黑體', 8, 'bold')).pack(side='left')
    ttk.Separator(f, orient='horizontal').pack(side='left', fill='x',
                                                expand=True, padx=(6, 0))


# ─────────── 主應用程式 ───────────

class WaterDropApp:
    CANVAS_W = 1100
    CANVAS_H = 618   # 16:9

    def __init__(self, root: tk.Tk, start_folder: Path = None):
        self.root = root
        self.root.title('水滴辨識系統')
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)

        # 播放狀態
        self.images           = []
        self.frame_idx        = 0
        self.fps              = DEFAULT_FPS
        self.paused           = True
        self.method           = 'B'
        self.thresh           = BINARIZE_THRESH
        self.manual_rois      = []
        self.draw_mode        = None   # None / 'replace' / 'add'
        self.prev_hist        = None
        self.wet_frames       = []
        self._after_id        = None
        self._frame_data      = {}   # {frame_idx: {name,drops,is_wet,chi2}}
        self._export_cancel   = False

        # 框選拖曳
        self._drag_start      = None
        self._drag_rect_id    = None

        # 縮放 / 平移
        self._zoom            = 1.0
        self._pan             = [0, 0]   # 從置中位置的偏移 (像素)
        self._base_scale      = 1.0      # fit-to-canvas 縮放比
        self._img_scale       = 1.0      # 有效縮放比 = _base_scale * _zoom
        self._img_offset      = (0, 0)   # 圖片在 canvas 上的左上角
        self._cur_img_wh      = (1280, 720)
        self._pan_start       = None
        self._pan_origin      = None

        # 圖表歷史
        self.ch_data          = deque(maxlen=HIST_WINDOW)
        self.cn_data          = deque(maxlen=HIST_WINDOW)
        self._wet_spans       = []

        self._build_ui()
        self._init_charts()

        if start_folder and start_folder.is_dir():
            self._load_folder(start_folder)

    # ──────────────────────────────────────────
    # UI 建構
    # ──────────────────────────────────────────

    def _build_ui(self):
        title_bar = tk.Frame(self.root, bg=C_BG)
        title_bar.pack(fill='x', padx=12, pady=(8, 0))
        tk.Label(title_bar, text='水滴辨識系統',
                 bg=C_BG, fg=C_ACCENT,
                 font=('微軟正黑體', 16, 'bold')).pack(side='left')
        tk.Label(title_bar, text='Water Droplet Detection',
                 bg=C_BG, fg=C_MUTED,
                 font=('微軟正黑體', 10)).pack(side='left', padx=(10, 0))

        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill='both', expand=True, padx=8, pady=6)

        # 左側：影像 + 圖表
        left = tk.Frame(body, bg=C_BG)
        left.pack(side='left', fill='both', expand=True)

        # 影像 Canvas
        canvas_wrap = tk.Frame(left, bg='#1e2d3d', bd=1, relief='flat')
        canvas_wrap.pack()
        self.canvas = tk.Canvas(
            canvas_wrap, bg='#000', cursor='crosshair',
            width=self.CANVAS_W, height=self.CANVAS_H,
            highlightthickness=0
        )
        self.canvas.pack()

        # 左鍵框選
        self.canvas.bind('<ButtonPress-1>',   self._on_press)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Double-Button-1>', self._reset_zoom)

        # 右鍵平移
        self.canvas.bind('<ButtonPress-3>',   self._on_pan_start)
        self.canvas.bind('<B3-Motion>',       self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-3>', lambda e: setattr(self, '_pan_start', None))

        # 滾輪縮放
        self.canvas.bind('<MouseWheel>',      self._on_zoom)      # Windows
        self.canvas.bind('<Button-4>',        self._on_zoom)      # Linux scroll up
        self.canvas.bind('<Button-5>',        self._on_zoom)      # Linux scroll down

        self._placeholder = self.canvas.create_text(
            self.CANVAS_W // 2, self.CANVAS_H // 2,
            text='請選擇圖片資料夾開始分析',
            fill='#334155', font=('微軟正黑體', 16)
        )

        # 提示條（框選座標 + 縮放資訊）
        self._hint_var = tk.StringVar(value='')
        hint_bar = tk.Frame(left, bg='#1c1a0a')
        hint_bar.pack(fill='x')
        tk.Label(hint_bar, textvariable=self._hint_var,
                 bg='#1c1a0a', fg='#fbbf24',
                 font=('Consolas', 9), height=1, anchor='w').pack(side='left', padx=6)
        self._zoom_var = tk.StringVar(value='100%')
        tk.Label(hint_bar, textvariable=self._zoom_var,
                 bg='#1c1a0a', fg='#94a3b8',
                 font=('Consolas', 9)).pack(side='right', padx=8)

        # 圖表區
        chart_frame = tk.Frame(left, bg=C_CARD)
        chart_frame.pack(fill='both', expand=True, pady=(6, 0))
        self._build_charts(chart_frame)

        # 右側控制欄
        right = tk.Frame(body, bg=C_PANEL, width=230)
        right.pack(side='right', fill='y', padx=(8, 0))
        right.pack_propagate(False)
        self._build_controls(right)

        # 鍵盤綁定
        self.root.bind('<space>',  lambda e: self._toggle_play())
        self.root.bind('<d>',      lambda e: self._start_draw())
        self.root.bind('<a>',      lambda e: self._start_add())
        self.root.bind('<c>',      lambda e: self._clear_roi())
        self.root.bind('<m>',      lambda e: self._toggle_method())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('<plus>',   lambda e: self._adj_fps(1))
        self.root.bind('<equal>',  lambda e: self._adj_fps(1))
        self.root.bind('<minus>',  lambda e: self._adj_fps(-1))
        self.root.bind('<r>',      lambda e: self._reset_zoom())

    def _build_controls(self, parent):
        pad = dict(padx=10, pady=2)

        # 資料夾
        section_label(parent, '資料夾')
        folder_row = tk.Frame(parent, bg=C_PANEL)
        folder_row.pack(fill='x', **pad)
        self._folder_lbl = tk.Label(
            folder_row, text='尚未選擇', bg=C_CARD, fg=C_MUTED,
            font=('微軟正黑體', 8), wraplength=150, anchor='w', justify='left',
            padx=4, pady=2
        )
        self._folder_lbl.pack(side='left', fill='x', expand=True, padx=(0, 4))
        flat_btn(folder_row, '選擇', self._select_folder,
                 bg='#3b82f6', padx=8).pack(side='right')

        # 即時狀態
        section_label(parent, '即時狀態')
        status_card = tk.Frame(parent, bg=C_CARD)
        status_card.pack(fill='x', padx=10, pady=2)

        self._sv = {}
        rows = [
            ('幀數',   'frame',  C_TEXT),
            ('狀態',   'status', C_GREEN),
            ('水滴數', 'drops',  C_TEXT),
            ('ROI 數', 'rois',   C_TEXT),
            ('FPS',    'fps',    C_TEXT),
        ]
        for label, key, color in rows:
            r = tk.Frame(status_card, bg=C_CARD)
            r.pack(fill='x', padx=6, pady=2)
            tk.Label(r, text=label, bg=C_CARD, fg=C_MUTED,
                     font=('微軟正黑體', 8), width=6, anchor='w').pack(side='left')
            v = tk.StringVar(value='—')
            lbl = tk.Label(r, textvariable=v, bg=C_CARD, fg=color,
                           font=('微軟正黑體', 9, 'bold'), anchor='w')
            lbl.pack(side='left')
            self._sv[key] = (v, lbl)

        # 播放控制
        section_label(parent, '播放控制')
        play_row = tk.Frame(parent, bg=C_PANEL)
        play_row.pack(fill='x', **pad)
        self._play_btn = flat_btn(play_row, '▶  播放', self._toggle_play,
                                  bg=C_GREEN, padx=10)
        self._play_btn.pack(side='left', padx=(0, 6))
        flat_btn(play_row, '◀', self._prev_frame,
                 bg='#475569', padx=7).pack(side='left', padx=1)
        flat_btn(play_row, '▶', self._next_frame,
                 bg='#475569', padx=7).pack(side='left', padx=1)

        fps_row = tk.Frame(parent, bg=C_PANEL)
        fps_row.pack(fill='x', padx=10, pady=1)
        tk.Label(fps_row, text='FPS', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=4, anchor='w').pack(side='left')
        self._fps_var = tk.IntVar(value=self.fps)
        tk.Scale(fps_row, from_=1, to=30, orient='horizontal',
                 variable=self._fps_var,
                 command=lambda v: setattr(self, 'fps', int(float(v))),
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=True, length=155
                 ).pack(side='right')

        # ROI 框選
        section_label(parent, 'ROI 框選')
        roi_row = tk.Frame(parent, bg=C_PANEL)
        roi_row.pack(fill='x', **pad)
        self._draw_btn = flat_btn(roi_row, '✎ 框選', self._start_draw,
                                  bg=C_YELLOW, padx=8)
        self._draw_btn.pack(side='left', padx=(0, 3))
        flat_btn(roi_row, '＋ 追加', self._start_add,
                 bg=C_PURPLE, padx=8).pack(side='left', padx=3)
        flat_btn(roi_row, '✕ 清除', self._clear_roi,
                 bg=C_RED, padx=8).pack(side='left', padx=3)

        self._roi_list_var = tk.StringVar(value='（自動偵測模式）')
        tk.Label(parent, textvariable=self._roi_list_var,
                 bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 7), wraplength=210,
                 anchor='w', justify='left').pack(fill='x', padx=10)

        # 縮放控制
        section_label(parent, '影像縮放')
        zoom_row = tk.Frame(parent, bg=C_PANEL)
        zoom_row.pack(fill='x', **pad)
        flat_btn(zoom_row, '－', lambda: self._zoom_step(0.8),
                 bg='#475569', padx=8).pack(side='left', padx=(0, 2))
        flat_btn(zoom_row, '＋', lambda: self._zoom_step(1.25),
                 bg='#475569', padx=8).pack(side='left', padx=2)
        flat_btn(zoom_row, '重設', self._reset_zoom,
                 bg='#334155', padx=6).pack(side='left', padx=(6, 0))
        tk.Label(zoom_row, textvariable=self._zoom_var,
                 bg=C_PANEL, fg=C_ACCENT,
                 font=('Consolas', 9, 'bold')).pack(side='right')
        tk.Label(parent, text='滾輪縮放｜右鍵拖曳平移｜雙擊重設',
                 bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 7)).pack(anchor='w', padx=10)

        # 偵測方法
        section_label(parent, '偵測方法')
        self._method_var = tk.StringVar(value='B')
        for val, text in [('B', 'TopHat（推薦）'), ('A', 'Threshold')]:
            tk.Radiobutton(
                parent, text=text, variable=self._method_var, value=val,
                command=self._on_method_change,
                bg=C_PANEL, fg=C_TEXT, selectcolor='#334155',
                activebackground=C_PANEL,
                font=('微軟正黑體', 9)
            ).pack(anchor='w', padx=12)

        thresh_row = tk.Frame(parent, bg=C_PANEL)
        thresh_row.pack(fill='x', padx=10, pady=1)
        tk.Label(thresh_row, text='閾值', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=4, anchor='w').pack(side='left')
        self._thresh_var = tk.IntVar(value=self.thresh)
        tk.Scale(thresh_row, from_=10, to=250, orient='horizontal',
                 variable=self._thresh_var,
                 command=lambda v: setattr(self, 'thresh', int(float(v))),
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=True, length=155
                 ).pack(side='right')

        # 鍵盤快捷鍵
        section_label(parent, '快捷鍵')
        key_frame = tk.Frame(parent, bg=C_CARD)
        key_frame.pack(fill='x', padx=10, pady=2)
        for key, desc in [('SPACE', '播放/暫停'), ('D', '框選 ROI'),
                           ('A', '追加 ROI'),    ('C', '清除 ROI'),
                           ('M', '切換方法'),    ('R', '重設縮放')]:
            r = tk.Frame(key_frame, bg=C_CARD)
            r.pack(fill='x', padx=4, pady=1)
            tk.Label(r, text=key, bg='#1e293b', fg=C_ACCENT,
                     font=('Consolas', 8, 'bold'), width=6,
                     relief='flat', padx=2).pack(side='left', padx=(0, 6))
            tk.Label(r, text=desc, bg=C_CARD, fg=C_MUTED,
                     font=('微軟正黑體', 8)).pack(side='left')

        # WET 統計
        section_label(parent, 'WET 幀紀錄')
        self._wet_stat_var = tk.StringVar(value='共 0 幀')
        tk.Label(parent, textvariable=self._wet_stat_var,
                 bg=C_PANEL, fg='#f87171',
                 font=('微軟正黑體', 9, 'bold')).pack(anchor='w', padx=12)
        flat_btn(parent, '查看統計', self._show_statistics,
                 bg='#334155', padx=8, pady=3
                 ).pack(padx=10, pady=(4, 2), anchor='w')

        # 匯出
        section_label(parent, '結果匯出')
        exp_row = tk.Frame(parent, bg=C_PANEL)
        exp_row.pack(fill='x', padx=10, pady=2)
        flat_btn(exp_row, '📄 CSV', self._export_csv,
                 bg='#0ea5e9', padx=8).pack(side='left', padx=(0, 4))
        flat_btn(exp_row, '🎬 影片', self._export_video,
                 bg='#7c3aed', padx=8).pack(side='left')
        tk.Label(parent, text='影片以目前 ROI 設定輸出',
                 bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 7)).pack(anchor='w', padx=10)

    def _build_charts(self, parent):
        self._fig = Figure(figsize=(13, 2.6), facecolor=C_CARD, dpi=80)
        self._fig.subplots_adjust(left=0.04, right=0.99, top=0.87,
                                   bottom=0.12, wspace=0.38)

        def mk(pos, title):
            ax = self._fig.add_subplot(pos)
            ax.set_facecolor('#060d14')
            ax.set_title(title, color=C_TEXT, fontsize=7, pad=3)
            ax.tick_params(colors='#94a3b8', labelsize=6)
            for sp in ax.spines.values():
                sp.set_edgecolor('#2d3748')
            return ax

        self._ax_roi    = mk(151, 'ROI')
        self._ax_tophat = mk(152, 'TopHat')
        self._ax_binary = mk(153, 'Binary')
        self._ax_change = mk(154, 'Hist Delta')
        self._ax_count  = mk(155, 'Drop Count')

        self._ax_change.axhline(HIST_CHANGE_THRESH, color='#facc15',
                                 ls='--', lw=0.8, alpha=0.7)
        self._lc, = self._ax_change.plot([], [], color='#f87171', lw=1.2)
        self._ln, = self._ax_count.plot([],  [], color='#34d399', lw=1.2)

        fc = FigureCanvasTkAgg(self._fig, master=parent)
        fc.draw()
        fc.get_tk_widget().pack(fill='both', expand=True)
        self._fig_canvas = fc

    def _init_charts(self):
        blank = np.zeros((40, 60), dtype=np.uint8)
        for ax in (self._ax_roi, self._ax_tophat, self._ax_binary):
            ax.imshow(blank, cmap='gray', aspect='auto')
            ax.axis('off')
        self._fig_canvas.draw_idle()

    # ──────────────────────────────────────────
    # 縮放 / 平移
    # ──────────────────────────────────────────

    def _on_zoom(self, event):
        # Windows: event.delta；Linux: Button-4/5
        if hasattr(event, 'delta') and event.delta != 0:
            factor = 1.15 if event.delta > 0 else 1 / 1.15
        elif event.num == 4:
            factor = 1.15
        else:
            factor = 1 / 1.15

        new_zoom = max(0.5, min(8.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        img_w, img_h = self._cur_img_wh
        mx, my = event.x, event.y

        # 找出滑鼠指到的圖片座標（用目前的 offset + scale）
        ox, oy = self._img_offset
        eff = self._img_scale  # = base_scale * zoom
        px = (mx - ox) / eff
        py = (my - oy) / eff

        # 新的 effective scale
        new_eff = self._base_scale * new_zoom
        new_eff_w = int(img_w * new_eff)
        new_eff_h = int(img_h * new_eff)

        # 讓 (px, py) 仍對應到 (mx, my)
        new_ox = mx - px * new_eff
        new_oy = my - py * new_eff

        # 解出 pan：new_ox = (CW - new_eff_w)//2 + pan_x
        self._pan[0] = int(new_ox - (self.CANVAS_W - new_eff_w) // 2)
        self._pan[1] = int(new_oy - (self.CANVAS_H - new_eff_h) // 2)
        self._zoom = new_zoom
        self._zoom_var.set(f'{int(self._zoom * 100)}%')

        if self.images:
            self._render_current()

    def _zoom_step(self, factor):
        """按鈕觸發縮放（以影像中心為基準）"""
        new_zoom = max(0.5, min(8.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        # 等比縮放 pan
        ratio = new_zoom / self._zoom
        self._pan[0] = int(self._pan[0] * ratio)
        self._pan[1] = int(self._pan[1] * ratio)
        self._zoom = new_zoom
        self._zoom_var.set(f'{int(self._zoom * 100)}%')
        if self.images:
            self._render_current()

    def _reset_zoom(self, event=None):
        self._zoom = 1.0
        self._pan  = [0, 0]
        self._zoom_var.set('100%')
        if self.images:
            self._render_current()

    def _on_pan_start(self, event):
        self._pan_start  = (event.x, event.y)
        self._pan_origin = list(self._pan)

    def _on_pan_drag(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan = [self._pan_origin[0] + dx, self._pan_origin[1] + dy]
        if self.images:
            self._render_current()

    # ──────────────────────────────────────────
    # 框選（Canvas 滑鼠事件）
    # ──────────────────────────────────────────

    def _canvas_to_img(self, cx, cy):
        ox, oy = self._img_offset
        s = self._img_scale          # 已含 zoom
        return int((cx - ox) / s), int((cy - oy) / s)

    def _on_press(self, event):
        if self.draw_mode is None:
            return
        self._drag_start = (event.x, event.y)
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

    def _on_drag(self, event):
        if self.draw_mode is None or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline='#facc15', width=2, dash=(5, 3)
        )
        # 即時顯示框選座標
        ix0, iy0 = self._canvas_to_img(x0, y0)
        ix1, iy1 = self._canvas_to_img(event.x, event.y)
        w, h = self._cur_img_wh
        r = clamp_rect(ix0, iy0, ix1, iy1, w, h)
        mode_str = '取代' if self.draw_mode == 'replace' else '追加'
        self._hint_var.set(
            f'[{mode_str}]  '
            f'({r[0]}, {r[1]}) → ({r[2]}, {r[3]})  '
            f'尺寸: {r[2]-r[0]} × {r[3]-r[1]} px'
        )

    def _on_release(self, event):
        if self.draw_mode is None or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        if self._drag_rect_id:
            self.canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        w, h = self._cur_img_wh
        ix0, iy0 = self._canvas_to_img(x0, y0)
        ix1, iy1 = self._canvas_to_img(event.x, event.y)
        rect = clamp_rect(ix0, iy0, ix1, iy1, w, h)

        if abs(rect[2] - rect[0]) < 5 or abs(rect[3] - rect[1]) < 5:
            self._hint_var.set('')
            return

        if self.draw_mode == 'replace':
            self.manual_rois = [rect]
        else:
            self.manual_rois.append(rect)

        self.draw_mode = None
        self._hint_var.set(
            f'ROI 已儲存: ({rect[0]}, {rect[1]}) → ({rect[2]}, {rect[3]})  '
            f'尺寸: {rect[2]-rect[0]} × {rect[3]-rect[1]} px'
        )
        self._draw_btn.config(bg=C_YELLOW)
        self._refresh_roi_list()
        if self.images:
            self._render_current()

    # ──────────────────────────────────────────
    # 控制動作
    # ──────────────────────────────────────────

    def _load_folder(self, folder: Path):
        imgs = collect_images(folder)
        if not imgs:
            messagebox.showwarning('無圖片',
                                   f'資料夾中找不到支援的圖片格式：\n{folder}')
            return
        self.images      = imgs
        self.frame_idx   = 0
        # manual_rois 刻意不重置：跨資料夾保留框選座標
        self.prev_hist   = None
        self.wet_frames  = []
        self._frame_data = {}
        self.ch_data.clear()
        self.cn_data.clear()
        self._folder_lbl.config(text=folder.name, fg=C_TEXT)
        self.canvas.delete(self._placeholder)
        self._wet_stat_var.set('共 0 幀')
        self._hint_var.set('')
        self._refresh_roi_list()
        self._render_current()
        print(f"[載入] {len(imgs)} 張圖片  資料夾: {folder}")

    def _select_folder(self):
        path = filedialog.askdirectory(title='選擇圖片資料夾')
        if path:
            self._load_folder(Path(path))

    def _toggle_play(self):
        if not self.images:
            return
        # 若已在最後一幀且暫停 → 重頭播放
        if self.paused and self.frame_idx >= len(self.images) - 1:
            self.frame_idx = 0
            self.prev_hist = None
            self._render_current()

        self.paused = not self.paused
        if self.paused:
            self._play_btn.config(text='▶  播放', bg=C_GREEN)
            if self._after_id:
                self.root.after_cancel(self._after_id)
                self._after_id = None
        else:
            self._play_btn.config(text='⏸  暫停', bg=C_YELLOW)
            self._schedule_next()

    def _schedule_next(self):
        if self.paused or not self.images:
            return
        delay = max(1, int(1000 / max(self.fps, 1)))
        self._after_id = self.root.after(delay, self._advance_frame)

    def _advance_frame(self):
        if self.paused or not self.images:
            return
        # 到達最後一幀 → 停止並顯示統計
        if self.frame_idx >= len(self.images) - 1:
            self.paused = True
            self._after_id = None
            self._play_btn.config(text='▶  播放', bg=C_GREEN)
            self._show_statistics()
            return
        self.frame_idx += 1
        self._render_current()
        self._schedule_next()

    def _prev_frame(self):
        if not self.images:
            return
        self.frame_idx = (self.frame_idx - 1) % len(self.images)
        self._render_current()

    def _next_frame(self):
        if not self.images:
            return
        self.frame_idx = (self.frame_idx + 1) % len(self.images)
        self._render_current()

    def _start_draw(self):
        self.draw_mode = 'replace'
        self._hint_var.set('  拖曳選取 ROI（取代現有）')
        self._draw_btn.config(bg='#fbbf24')
        self.paused = True
        self._play_btn.config(text='▶  播放', bg=C_GREEN)
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _start_add(self):
        self.draw_mode = 'add'
        self._hint_var.set('  拖曳選取 ROI（追加至現有）')
        self._draw_btn.config(bg='#a78bfa')
        self.paused = True
        self._play_btn.config(text='▶  播放', bg=C_GREEN)
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _clear_roi(self):
        self.manual_rois = []
        self.draw_mode   = None
        self._hint_var.set('')
        self._draw_btn.config(bg=C_YELLOW)
        self._refresh_roi_list()
        if self.images:
            self._render_current()

    def _on_method_change(self):
        self.method = self._method_var.get()
        if self.images:
            self._render_current()

    def _toggle_method(self):
        self.method = 'A' if self.method == 'B' else 'B'
        self._method_var.set(self.method)
        if self.images:
            self._render_current()

    def _adj_fps(self, delta):
        self.fps = max(1, min(30, self.fps + delta))
        self._fps_var.set(self.fps)

    def _refresh_roi_list(self):
        if not self.manual_rois:
            self._roi_list_var.set('（自動偵測模式）')
        else:
            lines = [f'ROI {i+1}: ({r[0]},{r[1]})→({r[2]},{r[3]})'
                     for i, r in enumerate(self.manual_rois)]
            self._roi_list_var.set('\n'.join(lines))

    # ──────────────────────────────────────────
    # 匯出功能
    # ──────────────────────────────────────────

    def _export_csv(self):
        if not self._frame_data:
            messagebox.showwarning('無資料', '請先播放或瀏覽圖片後再匯出。')
            return
        out_path = filedialog.asksaveasfilename(
            title='儲存 CSV',
            defaultextension='.csv',
            filetypes=[('CSV 檔案', '*.csv')]
        )
        if not out_path:
            return
        try:
            with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(['幀編號', '檔案名稱', '水滴數', '狀態', 'chi2'])
                for fi in sorted(self._frame_data):
                    d = self._frame_data[fi]
                    w.writerow([
                        fi + 1,
                        d['name'],
                        d['drops'],
                        'WET' if d['is_wet'] else 'DRY',
                        f"{d['chi2']:.6f}",
                    ])
            messagebox.showinfo('完成', f'CSV 已儲存：\n{out_path}')
        except Exception as e:
            messagebox.showerror('錯誤', str(e))

    def _export_video(self):
        if not self.images:
            messagebox.showwarning('無資料', '請先載入圖片資料夾。')
            return

        out_path = filedialog.asksaveasfilename(
            title='儲存影片',
            defaultextension='.mp4',
            filetypes=[('MP4 影片', '*.mp4'), ('AVI 影片', '*.avi')]
        )
        if not out_path:
            return

        # 讀第一張取得解析度
        first = read_image_unicode(self.images[0])
        if first is None:
            messagebox.showerror('錯誤', '無法讀取第一張圖片。')
            return
        h, w = first.shape[:2]

        ext = Path(out_path).suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'XVID'))
        out_fps = max(self.fps, 1)

        # 進度視窗
        prog_win = tk.Toplevel(self.root)
        prog_win.title('輸出影片中…')
        prog_win.configure(bg=C_BG)
        prog_win.geometry('400x140')
        prog_win.resizable(False, False)
        prog_win.grab_set()

        tk.Label(prog_win, text='正在輸出影片，請稍候…',
                 bg=C_BG, fg=C_TEXT,
                 font=('微軟正黑體', 10)).pack(pady=(18, 6))

        prog_var = tk.DoubleVar(value=0)
        prog_bar = ttk.Progressbar(prog_win, variable=prog_var,
                                    maximum=100, length=340)
        prog_bar.pack(pady=4)

        self._prog_lbl_var = tk.StringVar(value='')
        tk.Label(prog_win, textvariable=self._prog_lbl_var,
                 bg=C_BG, fg=C_MUTED,
                 font=('微軟正黑體', 8)).pack()

        self._export_cancel = False
        cancel_btn = flat_btn(prog_win, '取消',
                              lambda: setattr(self, '_export_cancel', True),
                              bg=C_RED, pady=3)
        cancel_btn.pack(pady=6)

        def run():
            vw = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
            prev_hist_v = None
            total = len(self.images)

            for i, img_path in enumerate(self.images):
                if self._export_cancel:
                    break

                img = read_image_unicode(img_path)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rois = self.manual_rois if self.manual_rois else [auto_roi_rect(gray)]

                all_blobs = []
                for rect in rois:
                    rx1, ry1, rx2, ry2 = rect
                    roi_g = gray[ry1:ry2, rx1:rx2]
                    if roi_g.size == 0:
                        all_blobs.append([])
                        continue
                    if self.method == 'B':
                        binary, _ = detect_method_b(roi_g)
                    else:
                        binary = detect_method_a(roi_g, self.thresh)
                    all_blobs.append(find_blobs(binary))

                cnt = sum(len(b) for b in all_blobs)

                rx1, ry1, rx2, ry2 = rois[0]
                rh = gray[ry1:ry2, rx1:rx2]
                cur_h = compute_hist(rh if rh.size > 0 else gray)
                d_val = chi2_dist(prev_hist_v, cur_h) if prev_hist_v is not None else 0.0
                prev_hist_v = cur_h.copy()
                wet = (d_val > HIST_CHANGE_THRESH) or (cnt > 30)

                vis = draw_overlay(img, rois, all_blobs, wet, False, None)

                # 幀編號文字
                label = f"#{i+1}/{total}  drops={cnt}"
                cv2.putText(vis, label, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(vis, label, (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)

                vw.write(vis)

                pct = (i + 1) / total * 100
                self.root.after(0, prog_var.set, pct)
                self.root.after(0, self._prog_lbl_var.set,
                                f'幀 {i+1} / {total}  ({pct:.0f}%)')

            vw.release()
            cancelled = self._export_cancel

            def done():
                prog_win.destroy()
                if cancelled:
                    messagebox.showinfo('已取消', '影片輸出已中止。')
                else:
                    messagebox.showinfo('完成', f'影片已儲存：\n{out_path}')

            self.root.after(0, done)

        threading.Thread(target=run, daemon=True).start()

    # ──────────────────────────────────────────
    # 統計視窗
    # ──────────────────────────────────────────

    def _show_statistics(self):
        total     = len(self.images)
        analyzed  = len(self._frame_data)
        wet_count = len(self.wet_frames)
        dry_count = analyzed - wet_count
        wet_pct   = wet_count / analyzed * 100 if analyzed > 0 else 0.0

        drops_vals  = [v['drops'] for v in self._frame_data.values()]
        max_drops   = max(drops_vals) if drops_vals else 0
        avg_drops   = sum(drops_vals) / len(drops_vals) if drops_vals else 0.0

        win = tk.Toplevel(self.root)
        win.title('分析統計結果')
        win.configure(bg=C_BG)
        win.geometry('520x540')
        win.resizable(False, False)
        win.grab_set()

        tk.Label(win, text='分析完成 — 統計結果',
                 bg=C_BG, fg=C_ACCENT,
                 font=('微軟正黑體', 14, 'bold')).pack(pady=(16, 6))

        # 摘要卡
        card = tk.Frame(win, bg=C_CARD)
        card.pack(fill='x', padx=20, pady=4)

        def stat_row(label, value, color=C_TEXT):
            r = tk.Frame(card, bg=C_CARD)
            r.pack(fill='x', padx=12, pady=4)
            tk.Label(r, text=label, bg=C_CARD, fg=C_MUTED,
                     font=('微軟正黑體', 9), width=12, anchor='w').pack(side='left')
            tk.Label(r, text=value, bg=C_CARD, fg=color,
                     font=('微軟正黑體', 10, 'bold'), anchor='w').pack(side='left')

        stat_row('總幀數',   f'{total} 幀')
        stat_row('已分析',   f'{analyzed} 幀')
        stat_row('WET 幀',   f'{wet_count} 幀  ({wet_pct:.1f}%)',   '#f87171')
        stat_row('DRY 幀',   f'{dry_count} 幀  ({100-wet_pct:.1f}%)', '#4ade80')
        stat_row('最大水滴', f'{max_drops} 個')
        stat_row('平均水滴', f'{avg_drops:.1f} 個/幀')
        stat_row('偵測方法', 'TopHat' if self.method == 'B'
                             else f'Threshold ({self.thresh})')
        stat_row('ROI 模式', f'手動 {len(self.manual_rois)} 個'
                             if self.manual_rois else '自動')

        # WET 幀清單
        tk.Label(win, text='WET 幀清單:',
                 bg=C_BG, fg=C_MUTED,
                 font=('微軟正黑體', 9)).pack(anchor='w', padx=20, pady=(8, 2))

        list_wrap = tk.Frame(win, bg=C_CARD)
        list_wrap.pack(fill='both', expand=True, padx=20, pady=(0, 4))

        sb = tk.Scrollbar(list_wrap)
        sb.pack(side='right', fill='y')

        lb = tk.Listbox(list_wrap, bg=C_CARD, fg='#f87171',
                        font=('Consolas', 8),
                        selectbackground='#334155',
                        yscrollcommand=sb.set,
                        relief='flat', bd=0)
        lb.pack(fill='both', expand=True)
        sb.config(command=lb.yview)

        if self.wet_frames:
            for fi in sorted(self.wet_frames):
                d = self._frame_data.get(fi, {})
                lb.insert('end',
                    f"  #{fi+1:>4d}  drops={d.get('drops',0):>4d}  {d.get('name', self.images[fi].name)}")
        else:
            lb.insert('end', '  （無 WET 幀）')

        flat_btn(win, '關閉', win.destroy, bg='#475569').pack(pady=8)

    # ──────────────────────────────────────────
    # 渲染
    # ──────────────────────────────────────────

    def _render_current(self):
        if not self.images:
            return
        path = self.images[self.frame_idx]
        img  = read_image_unicode(path)
        if img is None:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape
        self._cur_img_wh = (img_w, img_h)

        rois = self.manual_rois if self.manual_rois else [auto_roi_rect(gray)]

        all_blobs   = []
        tophat_show = np.zeros((40, 60), np.uint8)
        binary_show = np.zeros((40, 60), np.uint8)
        roi_show    = gray[:40, :60]

        for rect in rois:
            x1, y1, x2, y2 = rect
            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                all_blobs.append([])
                continue
            if self.method == 'B':
                binary, tophat_img = detect_method_b(roi_gray)
                tophat_show = tophat_img
            else:
                binary = detect_method_a(roi_gray, self.thresh)
                tophat_show = roi_gray
            binary_show = binary
            roi_show    = roi_gray
            all_blobs.append(find_blobs(binary))

        total_count = sum(len(b) for b in all_blobs)

        # 記錄每幀完整資料
        self._frame_data[self.frame_idx] = {
            'name':   path.name,
            'drops':  total_count,
            'is_wet': False,        # 更新於後方 is_wet 計算後
            'chi2':   0.0,
        }

        # 直方圖分析
        x1, y1, x2, y2 = rois[0]
        roi_hist = gray[y1:y2, x1:x2]
        cur_hist = compute_hist(roi_hist if roi_hist.size > 0 else gray)
        d = chi2_dist(self.prev_hist, cur_hist) if self.prev_hist is not None else 0.0
        self.prev_hist = cur_hist.copy()
        is_wet = (d > HIST_CHANGE_THRESH) or (total_count > 30)

        # 更新完整幀資料
        self._frame_data[self.frame_idx]['is_wet'] = is_wet
        self._frame_data[self.frame_idx]['chi2']   = d

        if is_wet and self.frame_idx not in self.wet_frames:
            self.wet_frames.append(self.frame_idx)
            print(f"  [WET] 幀{self.frame_idx+1:>4d} {path.name}"
                  f"  drops={total_count}  chi2={d:.4f}")

        # 影像疊加
        vis = draw_overlay(img, rois, all_blobs,
                           is_wet, self.paused, self.draw_mode)

        # 計算縮放（base = fit-to-canvas，乘上 zoom）
        cw, ch = self.CANVAS_W, self.CANVAS_H
        self._base_scale = min(cw / img_w, ch / img_h)
        eff_scale = self._base_scale * self._zoom
        eff_w = int(img_w * eff_scale)
        eff_h = int(img_h * eff_scale)

        # 計算 offset（置中 + pan）
        ox = (cw - eff_w) // 2 + self._pan[0]
        oy = (ch - eff_h) // 2 + self._pan[1]
        self._img_scale  = eff_scale
        self._img_offset = (ox, oy)

        # 縮放圖片並貼到黑色 canvas
        resized = cv2.resize(vis, (eff_w, eff_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        canvas_img = Image.new('RGB', (cw, ch), (0, 0, 0))
        canvas_img.paste(pil, (ox, oy))  # PIL 自動裁切超出邊界的部分

        self._tk_img = ImageTk.PhotoImage(canvas_img)
        self.canvas.create_image(0, 0, anchor='nw', image=self._tk_img)

        # 更新右欄狀態
        self._sv['frame'][0].set(f"{self.frame_idx + 1} / {len(self.images)}")
        self._sv['drops'][0].set(str(total_count))
        self._sv['rois'][0].set(str(len(rois)))
        self._sv['fps'][0].set(str(self.fps))
        if is_wet:
            self._sv['status'][0].set('WET  水滴密集')
            self._sv['status'][1].config(fg='#f87171')
        else:
            self._sv['status'][0].set('DRY  正常')
            self._sv['status'][1].config(fg='#4ade80')
        self._wet_stat_var.set(f'共 {len(self.wet_frames)} 幀')

        # 更新圖表
        self._update_charts(roi_show, tophat_show, binary_show,
                            all_blobs[0] if all_blobs else [],
                            d, total_count, is_wet)

    def _update_charts(self, roi_gray, tophat_img, binary, blobs,
                       chi2, count, is_wet):
        for ax, img, title, cmap in [
            (self._ax_roi,    roi_gray,   'ROI',    'gray'),
            (self._ax_tophat, tophat_img, 'TopHat', 'inferno'),
        ]:
            ax.cla()
            ax.set_facecolor('#060d14')
            ax.set_title(title, color=C_TEXT, fontsize=7, pad=3)
            ax.imshow(img, cmap=cmap, aspect='auto')
            ax.axis('off')

        self._ax_binary.cla()
        self._ax_binary.set_facecolor('#060d14')
        self._ax_binary.set_title(f'Binary ({count} drops)',
                                   color=C_TEXT, fontsize=7, pad=3)
        bv = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        for b in blobs:
            r = max(3, int(np.sqrt(b['area'] / np.pi)) + 1)
            cv2.circle(bv, (int(b['cx']), int(b['cy'])), r, (255, 140, 0), 1)
        self._ax_binary.imshow(bv, aspect='auto')
        self._ax_binary.axis('off')

        self.ch_data.append(chi2)
        self.cn_data.append(count)
        yc = list(self.ch_data)
        yn = list(self.cn_data)

        self._lc.set_data(range(len(yc)), yc)
        self._ax_change.set_xlim(0, max(HIST_WINDOW, len(yc)))
        self._ax_change.set_ylim(
            0, max(max(yc) * 1.25 if yc else 0.05,
                   HIST_CHANGE_THRESH * 1.5, 0.05))
        for sp in self._wet_spans:
            try:
                sp.remove()
            except Exception:
                pass
        self._wet_spans = [
            self._ax_change.axvspan(i - .5, i + .5, color='#f87171', alpha=0.15)
            for i, v in enumerate(yc) if v > HIST_CHANGE_THRESH
        ]
        self._ax_change.set_title('Hist Delta', color=C_TEXT, fontsize=7, pad=3)
        self._ax_change.tick_params(colors='#94a3b8', labelsize=6)
        for sp in self._ax_change.spines.values():
            sp.set_edgecolor('#2d3748')

        self._ln.set_data(range(len(yn)), yn)
        for coll in list(self._ax_count.collections):
            try:
                coll.remove()
            except Exception:
                pass
        self._ax_count.fill_between(range(len(yn)), yn,
                                     color='#34d399', alpha=0.15)
        self._ax_count.set_xlim(0, max(HIST_WINDOW, len(yn)))
        self._ax_count.set_ylim(0, max(max(yn) * 1.3 if yn else 1, 5))
        self._ax_count.set_title('Drop Count', color=C_TEXT, fontsize=7, pad=3)
        self._ax_count.tick_params(colors='#94a3b8', labelsize=6)
        for sp in self._ax_count.spines.values():
            sp.set_edgecolor('#2d3748')

        self._fig_canvas.draw_idle()


# ─────────── 主程式進入點 ───────────

def main():
    parser = argparse.ArgumentParser(description='水滴辨識系統')
    parser.add_argument('folder', nargs='?', default=None,
                        help='圖片資料夾（支援中文路徑）')
    parser.add_argument('--fps',    type=float, default=DEFAULT_FPS)
    parser.add_argument('--thresh', type=int,   default=BINARIZE_THRESH)
    args = parser.parse_args()

    start_folder = None
    if args.folder:
        start_folder = Path(args.folder)
        if not start_folder.is_dir():
            print(f"[錯誤] 找不到資料夾：{start_folder}")
            sys.exit(1)

    root = tk.Tk()
    app = WaterDropApp(root, start_folder=start_folder)
    app.fps    = int(args.fps)
    app.thresh = args.thresh
    app._fps_var.set(app.fps)
    app._thresh_var.set(app.thresh)
    root.mainloop()


if __name__ == '__main__':
    main()
