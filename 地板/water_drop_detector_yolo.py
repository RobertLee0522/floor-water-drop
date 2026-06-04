"""
水滴辨識系統 — YOLO 版 GUI 介面
================================
用法：
  python water_drop_detector_yolo.py [圖片資料夾] [--model 模型路徑]

操作方式：
  滑鼠拖曳       : 框選 ROI（先點「框選」按鈕或按 D）
  滾輪           : 縮放影像
  右鍵拖曳       : 平移（縮放後使用）
  雙擊左鍵       : 重設縮放
  SPACE          : 播放 / 暫停
  D              : 進入框選模式
  A              : 追加 ROI
  C              : 清除所有 ROI
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
DEFAULT_CONF       = 0.25      # YOLO 信心度閾值
DEFAULT_IOU        = 0.45      # YOLO NMS IoU 閾值
WET_DROP_COUNT     = 5         # 超過此數量判定為 WET
HIST_WINDOW        = 60
HIST_CHANGE_THRESH = 0.25
ROW_THRESH_RATIO   = 1.3
ROI_PADDING        = 30
SUPPORTED_EXT      = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

DEFAULT_MODEL = str(
    Path(__file__).parent.parent /
    'runs' / 'train' / 'stealtheyes_yolov112' / 'weights' / 'best.pt'
)

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


# ─────────── 影像讀取工具 ───────────

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


def clamp_rect(x1, y1, x2, y2, W, H):
    x1, x2 = sorted([max(0, min(x1, W - 1)), max(0, min(x2, W - 1))])
    y1, y2 = sorted([max(0, min(y1, H - 1)), max(0, min(y2, H - 1))])
    return x1, y1, x2, y2


def compute_hist(gray: np.ndarray):
    h = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    s = h.sum()
    return h / s if s > 0 else h


def chi2_dist(h1, h2):
    eps = 1e-10
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)) / 2)


# ─────────── YOLO 推論 ───────────

def run_yolo(model, img_bgr: np.ndarray, roi_rect, conf: float, iou: float):
    """
    在 roi_rect 範圍內執行 YOLO 推論。
    回傳 (detections, conf_img)
      detections: list of {'x1','y1','x2','y2','conf','cls'} — 座標為全圖絕對像素
      conf_img  : uint8 灰階 confidence heatmap (roi 尺寸)
    """
    rx1, ry1, rx2, ry2 = roi_rect
    roi = img_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return [], np.zeros((40, 60), np.uint8)

    results = model(roi, conf=conf, iou=iou, verbose=False)
    detections = []
    h_roi, w_roi = roi.shape[:2]
    conf_map = np.zeros((h_roi, w_roi), np.float32)

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            c = float(box.conf[0].cpu())
            cls = int(box.cls[0].cpu())
            detections.append({
                'x1': bx1 + rx1, 'y1': by1 + ry1,
                'x2': bx2 + rx1, 'y2': by2 + ry1,
                'conf': c, 'cls': cls,
            })
            bx1c = max(0, bx1); by1c = max(0, by1)
            bx2c = min(w_roi, bx2); by2c = min(h_roi, by2)
            conf_map[by1c:by2c, bx1c:bx2c] = np.maximum(
                conf_map[by1c:by2c, bx1c:bx2c], c)

    conf_img = (conf_map * 255).astype(np.uint8)
    return detections, conf_img


def draw_overlay(frame, rois, all_dets, is_wet, paused, draw_mode):
    vis = frame.copy()
    h, w = vis.shape[:2]

    for idx, (rect, dets) in enumerate(zip(rois, all_dets)):
        rx1, ry1, rx2, ry2 = rect
        color = ROI_COLORS_BGR[idx % len(ROI_COLORS_BGR)]
        cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 2)
        label = f"ROI {idx+1}: {len(dets)} drops"
        cv2.putText(vis, label, (rx1 + 4, ry1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (rx1 + 4, ry1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        for d in dets:
            bx1, by1, bx2, by2 = d['x1'], d['y1'], d['x2'], d['y2']
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 200, 255), 1)
            cf_txt = f"{d['conf']:.2f}"
            cv2.putText(vis, cf_txt, (bx1, by1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 200, 255), 1, cv2.LINE_AA)

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

class WaterDropYOLOApp:
    CANVAS_W = 1100
    CANVAS_H = 618

    def __init__(self, root: tk.Tk, start_folder: Path = None,
                 model_path: str = DEFAULT_MODEL):
        self.root = root
        self.root.title('水滴辨識系統 — YOLO')
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)

        self.model       = None
        self.model_path  = model_path
        self.conf        = DEFAULT_CONF
        self.iou         = DEFAULT_IOU

        self.images      = []
        self.frame_idx   = 0
        self.fps         = DEFAULT_FPS
        self.paused      = True
        self.manual_rois = []
        self.draw_mode   = None
        self.prev_hist   = None
        self.wet_frames  = []
        self._after_id   = None
        self._frame_data = {}
        self._export_cancel = False

        self._drag_start   = None
        self._drag_rect_id = None

        self._zoom       = 1.0
        self._pan        = [0, 0]
        self._base_scale = 1.0
        self._img_scale  = 1.0
        self._img_offset = (0, 0)
        self._cur_img_wh = (1280, 720)
        self._pan_start  = None
        self._pan_origin = None

        self.ch_data   = deque(maxlen=HIST_WINDOW)
        self.cn_data   = deque(maxlen=HIST_WINDOW)
        self._wet_spans = []

        self._build_ui()
        self._init_charts()

        # 非同步載入模型，避免阻塞 UI
        threading.Thread(target=self._load_model, daemon=True).start()

        if start_folder and start_folder.is_dir():
            self._load_folder(start_folder)

    # ──────────────────────────────────────────
    # 模型載入
    # ──────────────────────────────────────────

    def _load_model(self, path: str = None):
        target = path or self.model_path
        self.root.after(0, self._sv['model'][0].set, '載入中…')
        try:
            from ultralytics import YOLO
            m = YOLO(target)
            self.model      = m
            self.model_path = target
            name = Path(target).parent.parent.name
            self.root.after(0, self._sv['model'][0].set, f'✔ {name}')
            self.root.after(0, self._sv['model'][1].config, {'fg': C_GREEN})
            print(f"[模型] 載入成功: {target}")
        except Exception as e:
            self.root.after(0, self._sv['model'][0].set, f'✘ 載入失敗')
            self.root.after(0, self._sv['model'][1].config, {'fg': C_RED})
            print(f"[模型] 載入失敗: {e}")

    def _select_model(self):
        path = filedialog.askopenfilename(
            title='選擇 YOLO 模型（.pt）',
            filetypes=[('PyTorch 模型', '*.pt'), ('所有檔案', '*.*')]
        )
        if path:
            threading.Thread(target=self._load_model,
                             args=(path,), daemon=True).start()

    # ──────────────────────────────────────────
    # UI 建構
    # ──────────────────────────────────────────

    def _build_ui(self):
        title_bar = tk.Frame(self.root, bg=C_BG)
        title_bar.pack(fill='x', padx=12, pady=(8, 0))
        tk.Label(title_bar, text='水滴辨識系統',
                 bg=C_BG, fg=C_ACCENT,
                 font=('微軟正黑體', 16, 'bold')).pack(side='left')
        tk.Label(title_bar, text='YOLO Detection',
                 bg=C_BG, fg=C_MUTED,
                 font=('微軟正黑體', 10)).pack(side='left', padx=(10, 0))

        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill='both', expand=True, padx=8, pady=6)

        left = tk.Frame(body, bg=C_BG)
        left.pack(side='left', fill='both', expand=True)

        canvas_wrap = tk.Frame(left, bg='#1e2d3d', bd=1, relief='flat')
        canvas_wrap.pack()
        self.canvas = tk.Canvas(
            canvas_wrap, bg='#000', cursor='crosshair',
            width=self.CANVAS_W, height=self.CANVAS_H,
            highlightthickness=0
        )
        self.canvas.pack()

        self.canvas.bind('<ButtonPress-1>',   self._on_press)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Double-Button-1>', self._reset_zoom)
        self.canvas.bind('<ButtonPress-3>',   self._on_pan_start)
        self.canvas.bind('<B3-Motion>',       self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-3>', lambda e: setattr(self, '_pan_start', None))
        self.canvas.bind('<MouseWheel>',      self._on_zoom)
        self.canvas.bind('<Button-4>',        self._on_zoom)
        self.canvas.bind('<Button-5>',        self._on_zoom)

        self._placeholder = self.canvas.create_text(
            self.CANVAS_W // 2, self.CANVAS_H // 2,
            text='請選擇圖片資料夾開始分析',
            fill='#334155', font=('微軟正黑體', 16)
        )

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

        chart_frame = tk.Frame(left, bg=C_CARD)
        chart_frame.pack(fill='both', expand=True, pady=(6, 0))
        self._build_charts(chart_frame)

        right = tk.Frame(body, bg=C_PANEL, width=230)
        right.pack(side='right', fill='y', padx=(8, 0))
        right.pack_propagate(False)
        self._build_controls(right)

        self.root.bind('<space>',  lambda e: self._toggle_play())
        self.root.bind('<d>',      lambda e: self._start_draw())
        self.root.bind('<a>',      lambda e: self._start_add())
        self.root.bind('<c>',      lambda e: self._clear_roi())
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

        # 模型
        section_label(parent, 'YOLO 模型')
        self._sv = {}
        model_card = tk.Frame(parent, bg=C_CARD)
        model_card.pack(fill='x', padx=10, pady=2)
        mr = tk.Frame(model_card, bg=C_CARD)
        mr.pack(fill='x', padx=6, pady=3)
        tk.Label(mr, text='狀態', bg=C_CARD, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=6, anchor='w').pack(side='left')
        mv = tk.StringVar(value='未載入')
        ml = tk.Label(mr, textvariable=mv, bg=C_CARD, fg=C_YELLOW,
                      font=('微軟正黑體', 9, 'bold'), anchor='w')
        ml.pack(side='left')
        self._sv['model'] = (mv, ml)
        flat_btn(parent, '選擇模型 (.pt)', self._select_model,
                 bg='#334155', padx=8, pady=3).pack(padx=10, pady=(2, 4), anchor='w')

        # 即時狀態
        section_label(parent, '即時狀態')
        status_card = tk.Frame(parent, bg=C_CARD)
        status_card.pack(fill='x', padx=10, pady=2)

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

        # YOLO 參數
        section_label(parent, 'YOLO 參數')
        conf_row = tk.Frame(parent, bg=C_PANEL)
        conf_row.pack(fill='x', padx=10, pady=1)
        tk.Label(conf_row, text='Conf', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=5, anchor='w').pack(side='left')
        self._conf_var = tk.DoubleVar(value=self.conf)

        def _on_conf(v):
            self.conf = round(float(v), 2)
        tk.Scale(conf_row, from_=0.05, to=0.95, resolution=0.05,
                 orient='horizontal', variable=self._conf_var,
                 command=_on_conf,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=True, length=150
                 ).pack(side='right')

        iou_row = tk.Frame(parent, bg=C_PANEL)
        iou_row.pack(fill='x', padx=10, pady=1)
        tk.Label(iou_row, text='IoU', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=5, anchor='w').pack(side='left')
        self._iou_var = tk.DoubleVar(value=self.iou)

        def _on_iou(v):
            self.iou = round(float(v), 2)
        tk.Scale(iou_row, from_=0.05, to=0.95, resolution=0.05,
                 orient='horizontal', variable=self._iou_var,
                 command=_on_iou,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=True, length=150
                 ).pack(side='right')

        wet_row = tk.Frame(parent, bg=C_PANEL)
        wet_row.pack(fill='x', padx=10, pady=1)
        tk.Label(wet_row, text='WET閾', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=5, anchor='w').pack(side='left')
        self._wet_thresh_var = tk.IntVar(value=WET_DROP_COUNT)
        tk.Scale(wet_row, from_=1, to=50, orient='horizontal',
                 variable=self._wet_thresh_var,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=True, length=150
                 ).pack(side='right')

        # 快捷鍵
        section_label(parent, '快捷鍵')
        key_frame = tk.Frame(parent, bg=C_CARD)
        key_frame.pack(fill='x', padx=10, pady=2)
        for key, desc in [('SPACE', '播放/暫停'), ('D', '框選 ROI'),
                           ('A', '追加 ROI'),    ('C', '清除 ROI'),
                           ('R', '重設縮放')]:
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
        tk.Label(parent, text='影片以目前 ROI / YOLO 設定輸出',
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
        self._ax_conf   = mk(152, 'Conf Map')
        self._ax_boxes  = mk(153, 'Detections')
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
        for ax in (self._ax_roi, self._ax_conf, self._ax_boxes):
            ax.imshow(blank, cmap='gray', aspect='auto')
            ax.axis('off')
        self._fig_canvas.draw_idle()

    # ──────────────────────────────────────────
    # 縮放 / 平移
    # ──────────────────────────────────────────

    def _on_zoom(self, event):
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
        ox, oy = self._img_offset
        eff = self._img_scale
        px = (mx - ox) / eff
        py = (my - oy) / eff

        new_eff = self._base_scale * new_zoom
        new_eff_w = int(img_w * new_eff)
        new_eff_h = int(img_h * new_eff)
        new_ox = mx - px * new_eff
        new_oy = my - py * new_eff

        self._pan[0] = int(new_ox - (self.CANVAS_W - new_eff_w) // 2)
        self._pan[1] = int(new_oy - (self.CANVAS_H - new_eff_h) // 2)
        self._zoom = new_zoom
        self._zoom_var.set(f'{int(self._zoom * 100)}%')
        if self.images:
            self._render_current()

    def _zoom_step(self, factor):
        new_zoom = max(0.5, min(8.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
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
        s = self._img_scale
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
                w.writerow(['幀編號', '檔案名稱', '水滴數', '狀態', 'chi2', '最高信心度'])
                for fi in sorted(self._frame_data):
                    d = self._frame_data[fi]
                    w.writerow([
                        fi + 1,
                        d['name'],
                        d['drops'],
                        'WET' if d['is_wet'] else 'DRY',
                        f"{d['chi2']:.6f}",
                        f"{d.get('max_conf', 0.0):.4f}",
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

        first = read_image_unicode(self.images[0])
        if first is None:
            messagebox.showerror('錯誤', '無法讀取第一張圖片。')
            return
        h, w = first.shape[:2]

        ext = Path(out_path).suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*('mp4v' if ext == '.mp4' else 'XVID'))
        out_fps = max(self.fps, 1)

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
        ttk.Progressbar(prog_win, variable=prog_var,
                        maximum=100, length=340).pack(pady=4)

        self._prog_lbl_var = tk.StringVar(value='')
        tk.Label(prog_win, textvariable=self._prog_lbl_var,
                 bg=C_BG, fg=C_MUTED,
                 font=('微軟正黑體', 8)).pack()

        self._export_cancel = False
        flat_btn(prog_win, '取消',
                 lambda: setattr(self, '_export_cancel', True),
                 bg=C_RED, pady=3).pack(pady=6)

        wet_thresh = self._wet_thresh_var.get()

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

                all_dets = []
                if self.model:
                    for rect in rois:
                        dets, _ = run_yolo(self.model, img, rect,
                                           self.conf, self.iou)
                        all_dets.append(dets)
                else:
                    all_dets = [[] for _ in rois]

                cnt = sum(len(d) for d in all_dets)

                rx1, ry1, rx2, ry2 = rois[0]
                rh = gray[ry1:ry2, rx1:rx2]
                cur_h = compute_hist(rh if rh.size > 0 else gray)
                d_val = chi2_dist(prev_hist_v, cur_h) if prev_hist_v is not None else 0.0
                prev_hist_v = cur_h.copy()
                wet = (d_val > HIST_CHANGE_THRESH) or (cnt >= wet_thresh)

                vis = draw_overlay(img, rois, all_dets, wet, False, None)

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

        drops_vals = [v['drops'] for v in self._frame_data.values()]
        max_drops  = max(drops_vals) if drops_vals else 0
        avg_drops  = sum(drops_vals) / len(drops_vals) if drops_vals else 0.0
        max_conf_vals = [v.get('max_conf', 0.0) for v in self._frame_data.values()]
        avg_conf   = sum(max_conf_vals) / len(max_conf_vals) if max_conf_vals else 0.0

        win = tk.Toplevel(self.root)
        win.title('分析統計結果')
        win.configure(bg=C_BG)
        win.geometry('520x560')
        win.resizable(False, False)
        win.grab_set()

        tk.Label(win, text='分析完成 — 統計結果',
                 bg=C_BG, fg=C_ACCENT,
                 font=('微軟正黑體', 14, 'bold')).pack(pady=(16, 6))

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
        stat_row('平均信心度', f'{avg_conf:.3f}')
        stat_row('Conf 閾值', f'{self.conf:.2f}')
        stat_row('IoU 閾值',  f'{self.iou:.2f}')
        stat_row('ROI 模式', f'手動 {len(self.manual_rois)} 個'
                             if self.manual_rois else '自動')

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
                    f"  #{fi+1:>4d}  drops={d.get('drops',0):>4d}"
                    f"  conf={d.get('max_conf',0):.3f}"
                    f"  {d.get('name', self.images[fi].name)}")
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
        wet_thresh = self._wet_thresh_var.get()

        all_dets  = []
        conf_show = np.zeros((40, 60), np.uint8)
        roi_show  = gray[:40, :60]
        max_conf  = 0.0

        if self.model is None:
            all_dets = [[] for _ in rois]
            self._hint_var.set('  模型尚未載入，請選擇 .pt 檔案')
        else:
            for rect in rois:
                dets, c_img = run_yolo(self.model, img, rect,
                                       self.conf, self.iou)
                all_dets.append(dets)
                conf_show = c_img
                x1, y1, x2, y2 = rect
                roi_show = gray[y1:y2, x1:x2]
                for d in dets:
                    max_conf = max(max_conf, d['conf'])

        total_count = sum(len(d) for d in all_dets)

        self._frame_data[self.frame_idx] = {
            'name':     path.name,
            'drops':    total_count,
            'is_wet':   False,
            'chi2':     0.0,
            'max_conf': max_conf,
        }

        x1, y1, x2, y2 = rois[0]
        roi_hist = gray[y1:y2, x1:x2]
        cur_hist = compute_hist(roi_hist if roi_hist.size > 0 else gray)
        d_val = chi2_dist(self.prev_hist, cur_hist) if self.prev_hist is not None else 0.0
        self.prev_hist = cur_hist.copy()
        is_wet = (d_val > HIST_CHANGE_THRESH) or (total_count >= wet_thresh)

        self._frame_data[self.frame_idx]['is_wet'] = is_wet
        self._frame_data[self.frame_idx]['chi2']   = d_val

        if is_wet and self.frame_idx not in self.wet_frames:
            self.wet_frames.append(self.frame_idx)
            print(f"  [WET] 幀{self.frame_idx+1:>4d} {path.name}"
                  f"  drops={total_count}  chi2={d_val:.4f}  max_conf={max_conf:.3f}")

        vis = draw_overlay(img, rois, all_dets, is_wet,
                           self.paused, self.draw_mode)

        cw, ch = self.CANVAS_W, self.CANVAS_H
        self._base_scale = min(cw / img_w, ch / img_h)
        eff_scale = self._base_scale * self._zoom
        eff_w = int(img_w * eff_scale)
        eff_h = int(img_h * eff_scale)

        ox = (cw - eff_w) // 2 + self._pan[0]
        oy = (ch - eff_h) // 2 + self._pan[1]
        self._img_scale  = eff_scale
        self._img_offset = (ox, oy)

        resized = cv2.resize(vis, (eff_w, eff_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        canvas_img = Image.new('RGB', (cw, ch), (0, 0, 0))
        canvas_img.paste(pil, (ox, oy))

        self._tk_img = ImageTk.PhotoImage(canvas_img)
        self.canvas.create_image(0, 0, anchor='nw', image=self._tk_img)

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

        self._update_charts(roi_show, conf_show, img, rois, all_dets,
                            d_val, total_count, is_wet)

    def _update_charts(self, roi_gray, conf_img, full_img,
                       rois, all_dets, chi2, count, is_wet):
        # ROI 灰階
        self._ax_roi.cla()
        self._ax_roi.set_facecolor('#060d14')
        self._ax_roi.set_title('ROI', color=C_TEXT, fontsize=7, pad=3)
        self._ax_roi.imshow(roi_gray, cmap='gray', aspect='auto')
        self._ax_roi.axis('off')

        # Confidence heatmap
        self._ax_conf.cla()
        self._ax_conf.set_facecolor('#060d14')
        self._ax_conf.set_title('Conf Map', color=C_TEXT, fontsize=7, pad=3)
        self._ax_conf.imshow(conf_img, cmap='hot', aspect='auto')
        self._ax_conf.axis('off')

        # Detection 預覽（第一個 ROI 的偵測框）
        self._ax_boxes.cla()
        self._ax_boxes.set_facecolor('#060d14')
        self._ax_boxes.set_title(f'Detections ({count})',
                                  color=C_TEXT, fontsize=7, pad=3)
        if rois and full_img is not None:
            rx1, ry1, rx2, ry2 = rois[0]
            crop = full_img[ry1:ry2, rx1:rx2].copy()
            for d in (all_dets[0] if all_dets else []):
                bx1 = d['x1'] - rx1; by1 = d['y1'] - ry1
                bx2 = d['x2'] - rx1; by2 = d['y2'] - ry1
                cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 200, 255), 1)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            self._ax_boxes.imshow(crop_rgb, aspect='auto')
        self._ax_boxes.axis('off')

        # Hist Delta
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

        # Drop Count
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
    parser = argparse.ArgumentParser(description='水滴辨識系統 — YOLO 版')
    parser.add_argument('folder', nargs='?', default=None,
                        help='圖片資料夾（支援中文路徑）')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='YOLO 模型路徑（.pt）')
    parser.add_argument('--fps',  type=float, default=DEFAULT_FPS)
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF)
    parser.add_argument('--iou',  type=float, default=DEFAULT_IOU)
    args = parser.parse_args()

    start_folder = None
    if args.folder:
        start_folder = Path(args.folder)
        if not start_folder.is_dir():
            print(f"[錯誤] 找不到資料夾：{start_folder}")
            sys.exit(1)

    root = tk.Tk()
    app = WaterDropYOLOApp(root, start_folder=start_folder,
                           model_path=args.model)
    app.fps  = int(args.fps)
    app.conf = args.conf
    app.iou  = args.iou
    app._fps_var.set(app.fps)
    app._conf_var.set(app.conf)
    app._iou_var.set(app.iou)
    root.mainloop()


if __name__ == '__main__':
    main()
