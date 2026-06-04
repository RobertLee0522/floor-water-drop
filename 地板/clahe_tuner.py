"""
CLAHE + 二值化 影像品質調整工具
=================================
即時預覽 CLAHE 效果與二值化結果，支援縮放平移、批次輸出。

操作方式：
  滑鼠滾輪       : 縮放（以滑鼠為基準點）
  右鍵拖曳       : 平移
  雙擊左鍵       : 重設縮放
  ← / →          : 上 / 下一張圖片
  S              : 儲存 CLAHE 結果
  R              : 重設縮放
  + / -          : 縮放
  ESC            : 離開

用法：
  python clahe_tuner.py [圖片路徑或資料夾]
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import sys
import threading

# ─── 常數 ───
SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

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

# ═══════════════════════════════════════
#  影像處理函式
# ═══════════════════════════════════════

def read_img(path: Path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def save_img(path: Path, img: np.ndarray) -> bool:
    ok, buf = cv2.imencode(path.suffix.lower(), img)
    if ok:
        buf.tofile(str(path))
    return ok

def collect_images(folder: Path):
    return sorted(
        [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXT],
        key=lambda p: p.name
    )

# ─── CLAHE ───

def apply_clahe(bgr: np.ndarray, clip: float, tile: int, channel: str) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    if channel == 'L':
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    elif channel == 'Y':
        ycr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycr)
        return cv2.cvtColor(cv2.merge([clahe.apply(y), cr, cb]), cv2.COLOR_YCrCb2BGR)
    elif channel == 'Gray':
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    else:  # RGB-each
        b, g, r = cv2.split(bgr)
        return cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])

# ─── 二值化（整合自 water_drop_detector.py）───

def binarize_tophat(gray: np.ndarray, sigma: float = 1.5,
                    kernel_size: int = 61):
    """TopHat 形態學增強後二值化（偵測亮斑水滴最佳）"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    th = max(float(tophat.mean() + sigma * tophat.std()), 8.0)
    _, binary = cv2.threshold(tophat, int(th), 255, cv2.THRESH_BINARY)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ke, iterations=1)
    return binary, tophat

def binarize_threshold(gray: np.ndarray, thresh: int) -> np.ndarray:
    """固定閾值 + Gaussian 模糊"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    return binary

def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu 自動閾值"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)

def binarize_adaptive(gray: np.ndarray, block_size: int = 11,
                      c_val: int = 2) -> np.ndarray:
    """自適應 Gaussian 閾值"""
    bs = block_size | 1          # 確保為奇數
    bs = max(3, bs)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, bs, c_val
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)

def find_blobs(binary: np.ndarray, min_area: int, max_area: int):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            blobs.append({'cx': centroids[i][0],
                          'cy': centroids[i][1],
                          'area': area})
    return blobs

def draw_binary_vis(binary: np.ndarray, blobs: list) -> np.ndarray:
    """二值化 BGR 視覺化（白底 + 橘色 blob 圈）"""
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for b in blobs:
        r  = max(3, int(np.sqrt(b['area'] / np.pi)) + 1)
        cx = int(b['cx']); cy = int(b['cy'])
        cv2.circle(vis, (cx, cy), r,  (0, 140, 255), 1)
        cv2.circle(vis, (cx, cy), 2,  (0, 140, 255), -1)
    return vis

# ─── 指標 ───

def calc_metrics(orig_gray: np.ndarray, clahe_gray: np.ndarray) -> dict:
    def contrast(g):   return float(g.std())
    def brightness(g): return float(g.mean())
    def sharpness(g):  return float(cv2.Laplacian(g, cv2.CV_64F).var())
    def entropy(g):
        h = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
        h = h[h > 0] / h.sum()
        return float(-np.sum(h * np.log2(h)))
    return {
        'Contrast Δ':   contrast(clahe_gray)   - contrast(orig_gray),
        'Brightness Δ': brightness(clahe_gray)  - brightness(orig_gray),
        'Sharpness Δ':  sharpness(clahe_gray)   - sharpness(orig_gray),
        'Entropy Δ':    entropy(clahe_gray)      - entropy(orig_gray),
    }

# ═══════════════════════════════════════
#  UI 小工具
# ═══════════════════════════════════════

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
    ttk.Separator(f, orient='horizontal').pack(
        side='left', fill='x', expand=True, padx=(6, 0))

# ═══════════════════════════════════════
#  主 App
# ═══════════════════════════════════════

class CLAHETuner:
    CW, CH = 1280, 720   # canvas 尺寸（比原本更大）

    def __init__(self, root: tk.Tk, init_path: Path = None):
        self.root = root
        self.root.title('CLAHE + 二值化 影像品質調整工具')
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)

        # 影像狀態
        self._orig        = None   # BGR
        self._clahe_res   = None   # BGR
        self._binary_res  = None   # BGR (視覺化)
        self._blob_count  = 0
        self._base_display = None  # 合成顯示圖（CW × CH，BGR）

        self._cur_path  = None
        self._images: list = []
        self._img_idx   = 0
        self._after_id  = None

        # 縮放 / 平移
        self._zoom       = 1.0
        self._pan        = [0, 0]
        self._pan_start  = None
        self._pan_origin = None

        self._build_ui()

        if init_path:
            if init_path.is_dir():
                self._load_folder(init_path)
            elif init_path.is_file():
                self._load_file(init_path)

    # ══════════════════════════════════
    #  建構 UI
    # ══════════════════════════════════

    def _build_ui(self):
        # 標題列
        tb = tk.Frame(self.root, bg=C_BG)
        tb.pack(fill='x', padx=12, pady=(8, 0))
        tk.Label(tb, text='CLAHE + 二值化',
                 bg=C_BG, fg=C_ACCENT,
                 font=('微軟正黑體', 16, 'bold')).pack(side='left')
        tk.Label(tb, text='影像品質調整  ·  即時預覽',
                 bg=C_BG, fg=C_MUTED,
                 font=('微軟正黑體', 9)).pack(side='left', padx=(10, 0))

        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill='both', expand=True, padx=8, pady=6)

        # ── 左側：Canvas + 資訊列 ──
        left = tk.Frame(body, bg=C_BG)
        left.pack(side='left', fill='both', expand=True)

        wrap = tk.Frame(left, bg='#1e2d3d', bd=1, relief='flat')
        wrap.pack()
        self.canvas = tk.Canvas(wrap, bg='#000',
                                width=self.CW, height=self.CH,
                                highlightthickness=0)
        self.canvas.pack()

        # Canvas 事件
        self.canvas.bind('<MouseWheel>',      self._on_zoom)
        self.canvas.bind('<Button-4>',        self._on_zoom)   # Linux 向上
        self.canvas.bind('<Button-5>',        self._on_zoom)   # Linux 向下
        self.canvas.bind('<ButtonPress-3>',   self._on_pan_start)
        self.canvas.bind('<B3-Motion>',       self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-3>',
                         lambda e: setattr(self, '_pan_start', None))
        self.canvas.bind('<Double-Button-1>', lambda e: self._reset_zoom())

        self._placeholder = self.canvas.create_text(
            self.CW // 2, self.CH // 2,
            text='請開啟圖片或資料夾',
            fill='#334155', font=('微軟正黑體', 16))

        # 資訊列（下方）
        info_bar = tk.Frame(left, bg='#1c1a0a')
        info_bar.pack(fill='x')
        self._info_var = tk.StringVar(value='')
        tk.Label(info_bar, textvariable=self._info_var,
                 bg='#1c1a0a', fg='#fbbf24',
                 font=('Consolas', 9), height=1, anchor='w'
                 ).pack(side='left', padx=6)
        self._zoom_var = tk.StringVar(value='100%')
        tk.Label(info_bar, textvariable=self._zoom_var,
                 bg='#1c1a0a', fg='#94a3b8',
                 font=('Consolas', 9, 'bold')).pack(side='right', padx=8)

        # 指標列
        mbar = tk.Frame(left, bg=C_CARD)
        mbar.pack(fill='x', pady=(4, 0))
        # Blob 數量
        bf = tk.Frame(mbar, bg=C_CARD)
        bf.pack(side='left', padx=14, pady=4)
        tk.Label(bf, text='Blobs', bg=C_CARD, fg=C_MUTED,
                 font=('微軟正黑體', 7)).pack()
        self._blob_var = tk.StringVar(value='—')
        tk.Label(bf, textvariable=self._blob_var, bg=C_CARD,
                 fg='#f97316', font=('Consolas', 9, 'bold')).pack()
        # 4 個影像指標
        self._metric_vars = {}
        for key in ['Contrast Δ', 'Brightness Δ', 'Sharpness Δ', 'Entropy Δ']:
            f = tk.Frame(mbar, bg=C_CARD)
            f.pack(side='left', padx=14, pady=4)
            tk.Label(f, text=key, bg=C_CARD, fg=C_MUTED,
                     font=('微軟正黑體', 7)).pack()
            v = tk.StringVar(value='—')
            tk.Label(f, textvariable=v, bg=C_CARD, fg=C_ACCENT,
                     font=('Consolas', 9, 'bold')).pack()
            self._metric_vars[key] = v

        # ── 右側：可捲動控制欄 ──
        right_outer = tk.Frame(body, bg=C_PANEL, width=260)
        right_outer.pack(side='right', fill='y', padx=(8, 0))
        right_outer.pack_propagate(False)
        self._build_scrollable_controls(right_outer)

        # 鍵盤快捷鍵
        self.root.bind('<Left>',   lambda e: self._step(-1))
        self.root.bind('<Right>',  lambda e: self._step(1))
        self.root.bind('<s>',      lambda e: self._save_clahe())
        self.root.bind('<r>',      lambda e: self._reset_zoom())
        self.root.bind('<plus>',   lambda e: self._zoom_step(1.25))
        self.root.bind('<equal>',  lambda e: self._zoom_step(1.25))
        self.root.bind('<minus>',  lambda e: self._zoom_step(0.8))
        self.root.bind('<Escape>', lambda e: self.root.quit())

    # ── 可捲動框架 ──

    def _build_scrollable_controls(self, outer):
        ctrl_canvas = tk.Canvas(outer, bg=C_PANEL, highlightthickness=0)
        scrollbar   = ttk.Scrollbar(outer, orient='vertical',
                                    command=ctrl_canvas.yview)
        self._sf = tk.Frame(ctrl_canvas, bg=C_PANEL)

        self._sf.bind('<Configure>',
            lambda e: ctrl_canvas.configure(
                scrollregion=ctrl_canvas.bbox('all')))

        ctrl_canvas.create_window((0, 0), window=self._sf, anchor='nw')
        ctrl_canvas.configure(yscrollcommand=scrollbar.set)
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        def _mw(event):
            if hasattr(event, 'delta') and event.delta:
                ctrl_canvas.yview_scroll(-1 * (event.delta // 120), 'units')
            elif event.num == 4:
                ctrl_canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                ctrl_canvas.yview_scroll(1, 'units')

        for w in (ctrl_canvas, self._sf):
            w.bind('<MouseWheel>', _mw)
            w.bind('<Button-4>',  _mw)
            w.bind('<Button-5>',  _mw)

        self._build_controls(self._sf)

    # ── 控制內容 ──

    def _build_controls(self, p):
        pad = dict(padx=10, pady=2)

        # ── 開啟 ──
        section_label(p, '開啟')
        row = tk.Frame(p, bg=C_PANEL)
        row.pack(fill='x', **pad)
        flat_btn(row, '🖼 圖片', self._open_file,
                 bg='#3b82f6', padx=8).pack(side='left', padx=(0, 4))
        flat_btn(row, '📁 資料夾', self._open_folder,
                 bg='#475569', padx=8).pack(side='left')

        self._file_lbl = tk.Label(p, text='尚未開啟',
                                   bg=C_CARD, fg=C_MUTED,
                                   font=('微軟正黑體', 7), wraplength=228,
                                   anchor='w', justify='left', padx=4, pady=2)
        self._file_lbl.pack(fill='x', padx=10, pady=(2, 0))

        # ── 瀏覽 ──
        section_label(p, '瀏覽')
        nav = tk.Frame(p, bg=C_PANEL)
        nav.pack(fill='x', **pad)
        flat_btn(nav, '◀', lambda: self._step(-1),
                 bg='#475569', padx=12).pack(side='left', padx=(0, 4))
        flat_btn(nav, '▶', lambda: self._step(1),
                 bg='#475569', padx=12).pack(side='left')
        self._idx_var = tk.StringVar(value='—')
        tk.Label(p, textvariable=self._idx_var,
                 bg=C_PANEL, fg=C_MUTED,
                 font=('Consolas', 8)).pack(anchor='w', padx=12)

        # ── 顯示模式 ──
        section_label(p, '顯示模式')
        self._view_var = tk.StringVar(value='OC')
        for text, val in [
            ('原始  |  CLAHE',              'OC'),
            ('CLAHE  |  二值化',             'CB'),
            ('原始  |  CLAHE  |  二值化',   'OCB'),
        ]:
            tk.Radiobutton(
                p, text=text, variable=self._view_var, value=val,
                command=self._on_view_change,
                bg=C_PANEL, fg=C_TEXT, selectcolor='#334155',
                activebackground=C_PANEL,
                font=('微軟正黑體', 9)
            ).pack(anchor='w', padx=14)

        # ── 縮放 ──
        section_label(p, '影像縮放')
        zr = tk.Frame(p, bg=C_PANEL)
        zr.pack(fill='x', **pad)
        flat_btn(zr, '－', lambda: self._zoom_step(0.8),
                 bg='#475569', padx=9).pack(side='left', padx=(0, 2))
        flat_btn(zr, '＋', lambda: self._zoom_step(1.25),
                 bg='#475569', padx=9).pack(side='left', padx=2)
        flat_btn(zr, '重設', self._reset_zoom,
                 bg='#334155', padx=6).pack(side='left', padx=(6, 0))
        tk.Label(zr, textvariable=self._zoom_var,
                 bg=C_PANEL, fg=C_ACCENT,
                 font=('Consolas', 9, 'bold')).pack(side='right')
        tk.Label(p, text='滾輪縮放　右鍵拖曳平移　雙擊重設',
                 bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 7)).pack(anchor='w', padx=10)

        # ── CLAHE 參數 ──
        section_label(p, 'CLAHE 參數')

        def _slider_row(parent, label, lbl_widget_ref, unit=''):
            r = tk.Frame(parent, bg=C_PANEL)
            r.pack(fill='x', padx=10, pady=1)
            tk.Label(r, text=label, bg=C_PANEL, fg=C_MUTED,
                     font=('微軟正黑體', 8), width=10, anchor='w').pack(side='left')
            lbl = tk.Label(r, text='', bg=C_PANEL, fg=C_ACCENT,
                           font=('Consolas', 9, 'bold'), width=5)
            lbl.pack(side='right')
            lbl_widget_ref.append(lbl)
            return r

        self._clip_lbl_ref = []
        _slider_row(p, 'Clip Limit', self._clip_lbl_ref)
        self._clip_lbl = self._clip_lbl_ref[0]
        self._clip_lbl.config(text='2.0')
        self._clip_var = tk.DoubleVar(value=2.0)
        tk.Scale(p, from_=0.5, to=20.0, resolution=0.5,
                 orient='horizontal', variable=self._clip_var,
                 command=self._on_clip,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        self._tile_lbl_ref = []
        _slider_row(p, 'Tile Size', self._tile_lbl_ref)
        self._tile_lbl = self._tile_lbl_ref[0]
        self._tile_lbl.config(text='8×8')
        self._tile_var = tk.IntVar(value=8)
        tk.Scale(p, from_=2, to=64, resolution=2,
                 orient='horizontal', variable=self._tile_var,
                 command=self._on_tile,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        section_label(p, '色彩空間')
        self._channel_var = tk.StringVar(value='L')
        for text, val in [('L — LAB 亮度（推薦）', 'L'),
                           ('Y — YCrCb 亮度',      'Y'),
                           ('Gray — 灰階',         'Gray'),
                           ('RGB — 各通道獨立',    'RGB-each')]:
            tk.Radiobutton(
                p, text=text, variable=self._channel_var, value=val,
                command=self._schedule_apply,
                bg=C_PANEL, fg=C_TEXT, selectcolor='#334155',
                activebackground=C_PANEL, font=('微軟正黑體', 9)
            ).pack(anchor='w', padx=14)

        # ── 二值化方法 ──
        section_label(p, '二值化方法')
        self._bin_method_var = tk.StringVar(value='tophat')
        for text, val in [('TopHat（推薦水滴）', 'tophat'),
                           ('固定閾值',           'threshold'),
                           ('Otsu 自動閾值',      'otsu'),
                           ('自適應閾值',         'adaptive')]:
            tk.Radiobutton(
                p, text=text, variable=self._bin_method_var, value=val,
                command=self._schedule_apply,
                bg=C_PANEL, fg=C_TEXT, selectcolor='#334155',
                activebackground=C_PANEL, font=('微軟正黑體', 9)
            ).pack(anchor='w', padx=14)

        # ── 二值化參數 ──
        section_label(p, '二值化參數')

        # 固定閾值
        r1 = tk.Frame(p, bg=C_PANEL); r1.pack(fill='x', padx=10, pady=1)
        tk.Label(r1, text='閾值', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=10, anchor='w').pack(side='left')
        self._thresh_lbl = tk.Label(r1, text='128', bg=C_PANEL, fg=C_ACCENT,
                                     font=('Consolas', 9, 'bold'), width=5)
        self._thresh_lbl.pack(side='right')
        self._thresh_var = tk.IntVar(value=128)
        tk.Scale(p, from_=10, to=250, orient='horizontal',
                 variable=self._thresh_var, command=self._on_thresh,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        # TopHat sigma
        r2 = tk.Frame(p, bg=C_PANEL); r2.pack(fill='x', padx=10, pady=1)
        tk.Label(r2, text='TopHat σ', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=10, anchor='w').pack(side='left')
        self._sigma_lbl = tk.Label(r2, text='1.5', bg=C_PANEL, fg=C_ACCENT,
                                    font=('Consolas', 9, 'bold'), width=5)
        self._sigma_lbl.pack(side='right')
        self._sigma_var = tk.DoubleVar(value=1.5)
        tk.Scale(p, from_=0.5, to=8.0, resolution=0.5,
                 orient='horizontal', variable=self._sigma_var,
                 command=self._on_sigma,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        # Adaptive block size
        r3 = tk.Frame(p, bg=C_PANEL); r3.pack(fill='x', padx=10, pady=1)
        tk.Label(r3, text='Block Size', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=10, anchor='w').pack(side='left')
        self._bsize_lbl = tk.Label(r3, text='11', bg=C_PANEL, fg=C_ACCENT,
                                    font=('Consolas', 9, 'bold'), width=5)
        self._bsize_lbl.pack(side='right')
        self._bsize_var = tk.IntVar(value=11)
        tk.Scale(p, from_=3, to=51, resolution=2,
                 orient='horizontal', variable=self._bsize_var,
                 command=self._on_bsize,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        # TopHat kernel
        r4 = tk.Frame(p, bg=C_PANEL); r4.pack(fill='x', padx=10, pady=1)
        tk.Label(r4, text='TH Kernel', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=10, anchor='w').pack(side='left')
        self._kernel_lbl = tk.Label(r4, text='61', bg=C_PANEL, fg=C_ACCENT,
                                     font=('Consolas', 9, 'bold'), width=5)
        self._kernel_lbl.pack(side='right')
        self._kernel_var = tk.IntVar(value=61)
        tk.Scale(p, from_=11, to=151, resolution=10,
                 orient='horizontal', variable=self._kernel_var,
                 command=self._on_kernel,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        # ── Blob 面積過濾 ──
        section_label(p, 'Blob 面積過濾')
        r5 = tk.Frame(p, bg=C_PANEL); r5.pack(fill='x', padx=10, pady=1)
        tk.Label(r5, text='最小面積', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=10, anchor='w').pack(side='left')
        self._minarea_lbl = tk.Label(r5, text='5', bg=C_PANEL, fg=C_ACCENT,
                                      font=('Consolas', 9, 'bold'), width=5)
        self._minarea_lbl.pack(side='right')
        self._minarea_var = tk.IntVar(value=5)
        tk.Scale(p, from_=1, to=500, orient='horizontal',
                 variable=self._minarea_var, command=self._on_minarea,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        r6 = tk.Frame(p, bg=C_PANEL); r6.pack(fill='x', padx=10, pady=1)
        tk.Label(r6, text='最大面積', bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 8), width=10, anchor='w').pack(side='left')
        self._maxarea_lbl = tk.Label(r6, text='3000', bg=C_PANEL, fg=C_ACCENT,
                                      font=('Consolas', 9, 'bold'), width=5)
        self._maxarea_lbl.pack(side='right')
        self._maxarea_var = tk.IntVar(value=3000)
        tk.Scale(p, from_=100, to=20000, resolution=100,
                 orient='horizontal', variable=self._maxarea_var,
                 command=self._on_maxarea,
                 bg=C_PANEL, fg=C_TEXT, troughcolor='#374151',
                 highlightthickness=0, showvalue=False, length=235
                 ).pack(padx=10)

        # ── 二值化對象 ──
        section_label(p, '二值化套用於')
        self._bin_src_var = tk.StringVar(value='clahe')
        for text, val in [('CLAHE 結果（推薦）', 'clahe'),
                           ('原始影像',          'orig')]:
            tk.Radiobutton(
                p, text=text, variable=self._bin_src_var, value=val,
                command=self._schedule_apply,
                bg=C_PANEL, fg=C_TEXT, selectcolor='#334155',
                activebackground=C_PANEL, font=('微軟正黑體', 9)
            ).pack(anchor='w', padx=14)

        # ── 即時預覽 ──
        section_label(p, '預覽控制')
        self._live_var = tk.BooleanVar(value=True)
        tk.Checkbutton(p, text='即時預覽（拖動滑桿自動更新）',
                       variable=self._live_var,
                       bg=C_PANEL, fg=C_TEXT, selectcolor='#334155',
                       activebackground=C_PANEL,
                       font=('微軟正黑體', 9)).pack(anchor='w', padx=14)
        flat_btn(p, '▶ 套用', self._apply,
                 bg=C_GREEN, padx=10).pack(padx=10, pady=(4, 2), anchor='w')

        # ── 儲存 ──
        section_label(p, '儲存')
        sr1 = tk.Frame(p, bg=C_PANEL)
        sr1.pack(fill='x', **pad)
        flat_btn(sr1, '💾 CLAHE', self._save_clahe,
                 bg='#0ea5e9', padx=8).pack(side='left', padx=(0, 4))
        flat_btn(sr1, '💾 Binary', self._save_binary,
                 bg='#6366f1', padx=8).pack(side='left')
        flat_btn(p, '📦 批次輸出（CLAHE + Binary）', self._batch_export,
                 bg='#7c3aed', padx=6, font_size=8
                 ).pack(padx=10, pady=(4, 2), fill='x')
        tk.Label(p, text='批次對資料夾所有圖片套用並輸出',
                 bg=C_PANEL, fg=C_MUTED,
                 font=('微軟正黑體', 7)).pack(anchor='w', padx=10)

        # ── 快捷鍵 ──
        section_label(p, '快捷鍵')
        kf = tk.Frame(p, bg=C_CARD)
        kf.pack(fill='x', padx=10, pady=(2, 12))
        for key, desc in [('← →', '上/下一張'),
                           ('+ -', '縮放'),
                           ('S',   '儲存 CLAHE'),
                           ('R',   '重設縮放'),
                           ('ESC', '離開')]:
            r = tk.Frame(kf, bg=C_CARD)
            r.pack(fill='x', padx=4, pady=1)
            tk.Label(r, text=key, bg='#1e293b', fg=C_ACCENT,
                     font=('Consolas', 8, 'bold'), width=5,
                     relief='flat', padx=2).pack(side='left', padx=(0, 6))
            tk.Label(r, text=desc, bg=C_CARD, fg=C_MUTED,
                     font=('微軟正黑體', 8)).pack(side='left')

    # ══════════════════════════════════
    #  縮放 / 平移
    # ══════════════════════════════════

    def _on_zoom(self, event):
        if hasattr(event, 'delta') and event.delta != 0:
            factor = 1.15 if event.delta > 0 else 1 / 1.15
        elif event.num == 4:
            factor = 1.15
        else:
            factor = 1 / 1.15
        self._do_zoom(factor, pivot=(event.x, event.y))

    def _zoom_step(self, factor):
        self._do_zoom(factor, pivot=None)

    def _do_zoom(self, factor, pivot=None):
        new_zoom = max(0.25, min(12.0, self._zoom * factor))
        if new_zoom == self._zoom or self._base_display is None:
            return

        if pivot is not None:
            mx, my = pivot
            # 目前 crop 起點（可為負值，表示影像比 canvas 小）
            cur_crop_x = (self.CW * self._zoom - self.CW) / 2 - self._pan[0]
            cur_crop_y = (self.CH * self._zoom - self.CH) / 2 - self._pan[1]
            # 滑鼠指向的原始影像座標
            img_x = (cur_crop_x + mx) / self._zoom
            img_y = (cur_crop_y + my) / self._zoom
            # 新 zoom 下，讓相同影像座標仍在 (mx, my) 處
            new_crop_x = img_x * new_zoom - mx
            new_crop_y = img_y * new_zoom - my
            self._pan[0] = int((self.CW * new_zoom - self.CW) / 2 - new_crop_x)
            self._pan[1] = int((self.CH * new_zoom - self.CH) / 2 - new_crop_y)
        else:
            # 以畫面中心縮放：等比縮放 pan
            ratio = new_zoom / self._zoom
            self._pan[0] = int(self._pan[0] * ratio)
            self._pan[1] = int(self._pan[1] * ratio)

        self._zoom = new_zoom
        self._zoom_var.set(f'{int(self._zoom * 100)}%')
        self._render_canvas()

    def _reset_zoom(self):
        self._zoom = 1.0
        self._pan  = [0, 0]
        self._zoom_var.set('100%')
        self._render_canvas()

    def _on_pan_start(self, event):
        self._pan_start  = (event.x, event.y)
        self._pan_origin = list(self._pan)

    def _on_pan_drag(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan = [self._pan_origin[0] + dx,
                     self._pan_origin[1] + dy]
        self._render_canvas()

    # ══════════════════════════════════
    #  參數回調
    # ══════════════════════════════════

    def _on_clip(self, v):
        self._clip_lbl.config(text=f'{float(v):.1f}')
        if self._live_var.get(): self._schedule_apply()

    def _on_tile(self, v):
        t = int(float(v))
        self._tile_lbl.config(text=f'{t}×{t}')
        if self._live_var.get(): self._schedule_apply()

    def _on_thresh(self, v):
        self._thresh_lbl.config(text=str(int(float(v))))
        if self._live_var.get(): self._schedule_apply()

    def _on_sigma(self, v):
        self._sigma_lbl.config(text=f'{float(v):.1f}')
        if self._live_var.get(): self._schedule_apply()

    def _on_bsize(self, v):
        self._bsize_lbl.config(text=str(int(float(v))))
        if self._live_var.get(): self._schedule_apply()

    def _on_kernel(self, v):
        k = int(float(v))
        k = k if k % 2 == 1 else k + 1   # 強制奇數
        self._kernel_lbl.config(text=str(k))
        if self._live_var.get(): self._schedule_apply()

    def _on_minarea(self, v):
        self._minarea_lbl.config(text=str(int(float(v))))
        if self._live_var.get(): self._schedule_apply()

    def _on_maxarea(self, v):
        self._maxarea_lbl.config(text=str(int(float(v))))
        if self._live_var.get(): self._schedule_apply()

    def _on_view_change(self):
        if self._base_display is not None:
            self._compose_and_render()

    def _schedule_apply(self):
        """100 ms 防抖，避免滑桿拖曳時頻繁重算"""
        if self._after_id:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(100, self._apply)

    # ══════════════════════════════════
    #  讀取圖片
    # ══════════════════════════════════

    def _open_file(self):
        path = filedialog.askopenfilename(
            title='開啟圖片',
            filetypes=[('圖片', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp'),
                       ('所有檔案', '*.*')]
        )
        if path:
            self._load_file(Path(path))

    def _open_folder(self):
        path = filedialog.askdirectory(title='開啟圖片資料夾')
        if path:
            self._load_folder(Path(path))

    def _load_file(self, path: Path):
        img = read_img(path)
        if img is None:
            messagebox.showerror('錯誤', f'無法讀取：{path}')
            return
        self._images   = [path]
        self._img_idx  = 0
        self._orig     = img
        self._cur_path = path
        self._file_lbl.config(text=path.name, fg=C_TEXT)
        self._idx_var.set('1 / 1')
        self.canvas.delete(self._placeholder)
        self._apply()

    def _load_folder(self, folder: Path):
        imgs = collect_images(folder)
        if not imgs:
            messagebox.showwarning('無圖片', f'找不到支援圖片：\n{folder}')
            return
        self._images  = imgs
        self._img_idx = 0
        self._load_index(0)

    def _load_index(self, idx: int):
        path = self._images[idx]
        img  = read_img(path)
        if img is None:
            return
        self._orig     = img
        self._cur_path = path
        self._img_idx  = idx
        self._file_lbl.config(text=path.name, fg=C_TEXT)
        self._idx_var.set(f'{idx + 1} / {len(self._images)}')
        self.canvas.delete(self._placeholder)
        self._apply()

    def _step(self, delta: int):
        if not self._images:
            return
        self._load_index((self._img_idx + delta) % len(self._images))

    # ══════════════════════════════════
    #  核心處理
    # ══════════════════════════════════

    def _apply(self):
        if self._orig is None:
            return

        clip    = self._clip_var.get()
        tile    = int(self._tile_var.get())
        channel = self._channel_var.get()
        method  = self._bin_method_var.get()
        thresh  = int(self._thresh_var.get())
        sigma   = float(self._sigma_var.get())
        bsize   = int(self._bsize_var.get())
        kernel  = int(self._kernel_var.get())
        kernel  = kernel if kernel % 2 == 1 else kernel + 1
        min_a   = int(self._minarea_var.get())
        max_a   = int(self._maxarea_var.get())
        bin_src = self._bin_src_var.get()

        # 1. CLAHE
        clahe_res = apply_clahe(self._orig, clip, tile, channel)
        self._clahe_res = clahe_res

        # 2. 二值化
        src_bgr  = clahe_res if bin_src == 'clahe' else self._orig
        src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

        if method == 'tophat':
            binary, _ = binarize_tophat(src_gray, sigma, kernel)
        elif method == 'threshold':
            binary = binarize_threshold(src_gray, thresh)
        elif method == 'otsu':
            binary = binarize_otsu(src_gray)
        else:
            binary = binarize_adaptive(src_gray, bsize)

        blobs = find_blobs(binary, min_a, max_a)
        self._blob_count = len(blobs)
        self._binary_res = draw_binary_vis(binary, blobs)

        # 3. 指標
        og = cv2.cvtColor(self._orig, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(clahe_res,  cv2.COLOR_BGR2GRAY)
        m  = calc_metrics(og, rg)
        for key, var in self._metric_vars.items():
            val = m[key]
            var.set(f'{"+" if val >= 0 else ""}{val:.2f}')
        self._blob_var.set(str(self._blob_count))

        self._compose_and_render()

    # ── 合成並排 ──

    def _compose_and_render(self):
        if self._orig is None:
            return
        self._base_display = self._compose(self._view_var.get())
        self._render_canvas()

    def _compose(self, mode: str) -> np.ndarray:
        """生成並排合成圖（BGR，CW × CH）"""
        if mode == 'OC':
            panels = [
                (self._orig,       'Original',          (180, 180, 180)),
                (self._clahe_res,  'CLAHE',              (80, 220, 255)),
            ]
        elif mode == 'CB':
            panels = [
                (self._clahe_res,  'CLAHE',                          (80, 220, 255)),
                (self._binary_res, f'Binary  blobs={self._blob_count}', (80, 255, 150)),
            ]
        else:  # OCB
            panels = [
                (self._orig,       'Original',                         (180, 180, 180)),
                (self._clahe_res,  'CLAHE',                            (80, 220, 255)),
                (self._binary_res, f'Binary  blobs={self._blob_count}', (80, 255, 150)),
            ]

        n = len(panels)
        panel_w = self.CW // n
        canvas = np.zeros((self.CH, self.CW, 3), np.uint8)

        h, w = self._orig.shape[:2]
        scale = min((panel_w - 4) / w, (self.CH - 4) / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        for i, (img, label, color) in enumerate(panels):
            if img is None:
                continue
            disp = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            resized = cv2.resize(disp, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = i * panel_w + (panel_w - nw) // 2
            oy = (self.CH - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized

            # 面板標籤
            lx = i * panel_w + 6
            cv2.putText(canvas, label, (lx, oy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (lx, oy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,   1, cv2.LINE_AA)
            # 尺寸標籤
            cv2.putText(canvas, f'{w}×{h}', (lx, oy + nh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

        # 分隔線
        for i in range(1, n):
            cv2.line(canvas, (i * panel_w, 0), (i * panel_w, self.CH),
                     (55, 55, 55), 1)
        return canvas

    # ── 渲染（含縮放平移）──

    def _render_canvas(self):
        if self._base_display is None:
            return

        base = self._base_display          # CW × CH BGR
        zoom = self._zoom

        if zoom == 1.0 and self._pan == [0, 0]:
            display = base
        else:
            new_w = max(1, int(self.CW * zoom))
            new_h = max(1, int(self.CH * zoom))
            zoomed = cv2.resize(base, (new_w, new_h),
                                interpolation=cv2.INTER_LINEAR)

            # crop 起點（可負，表示影像比 canvas 小）
            crop_x = int((new_w - self.CW) / 2 - self._pan[0])
            crop_y = int((new_h - self.CH) / 2 - self._pan[1])

            # 來源在 zoomed 中的有效範圍
            sx = max(0, crop_x)
            sy = max(0, crop_y)
            ex = min(new_w, crop_x + self.CW)
            ey = min(new_h, crop_y + self.CH)

            # 目標在 canvas 中的偏移
            dx = max(0, -crop_x)
            dy = max(0, -crop_y)

            display = np.zeros((self.CH, self.CW, 3), np.uint8)
            if ex > sx and ey > sy:
                pw, ph = ex - sx, ey - sy
                display[dy:dy + ph, dx:dx + pw] = zoomed[sy:sy + ph, sx:sx + pw]

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.create_image(0, 0, anchor='nw', image=self._tk_img)

        # 資訊列更新
        if self._orig is not None and self._cur_path:
            h, w = self._orig.shape[:2]
            self._info_var.set(
                f'{self._cur_path.name}  {w}×{h}  │  '
                f'Clip={self._clip_var.get():.1f}  '
                f'Tile={self._tile_var.get()}  │  '
                f'blobs={self._blob_count}'
            )

    # ══════════════════════════════════
    #  儲存
    # ══════════════════════════════════

    def _save_clahe(self):
        if self._clahe_res is None:
            messagebox.showwarning('無結果', '請先套用 CLAHE。')
            return
        out = filedialog.asksaveasfilename(
            title='儲存 CLAHE 結果',
            initialfile=f'{self._cur_path.stem}_clahe{self._cur_path.suffix}',
            defaultextension=self._cur_path.suffix,
            filetypes=[('圖片', '*.png *.jpg *.bmp *.tif')]
        )
        if out:
            if save_img(Path(out), self._clahe_res):
                messagebox.showinfo('完成', f'已儲存：\n{out}')
            else:
                messagebox.showerror('錯誤', '儲存失敗。')

    def _save_binary(self):
        if self._binary_res is None:
            messagebox.showwarning('無結果', '請先套用二值化。')
            return
        out = filedialog.asksaveasfilename(
            title='儲存 Binary 結果',
            initialfile=f'{self._cur_path.stem}_binary{self._cur_path.suffix}',
            defaultextension=self._cur_path.suffix,
            filetypes=[('圖片', '*.png *.jpg *.bmp *.tif')]
        )
        if out:
            if save_img(Path(out), self._binary_res):
                messagebox.showinfo('完成', f'已儲存：\n{out}')
            else:
                messagebox.showerror('錯誤', '儲存失敗。')

    def _batch_export(self):
        if not self._images:
            messagebox.showwarning('無圖片', '請先開啟資料夾。')
            return
        out_dir = filedialog.askdirectory(title='選擇輸出資料夾')
        if not out_dir:
            return
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 凍結目前參數
        clip    = self._clip_var.get()
        tile    = int(self._tile_var.get())
        channel = self._channel_var.get()
        method  = self._bin_method_var.get()
        thresh  = int(self._thresh_var.get())
        sigma   = float(self._sigma_var.get())
        bsize   = int(self._bsize_var.get())
        kernel  = int(self._kernel_var.get())
        kernel  = kernel if kernel % 2 == 1 else kernel + 1
        min_a   = int(self._minarea_var.get())
        max_a   = int(self._maxarea_var.get())
        bin_src = self._bin_src_var.get()

        pw = tk.Toplevel(self.root)
        pw.title('批次輸出中…')
        pw.configure(bg=C_BG)
        pw.geometry('400x130')
        pw.resizable(False, False)
        pw.grab_set()
        tk.Label(pw, text='批次套用 CLAHE + 二值化，請稍候…',
                 bg=C_BG, fg=C_TEXT, font=('微軟正黑體', 10)).pack(pady=(14, 4))
        pv = tk.DoubleVar(value=0)
        ttk.Progressbar(pw, variable=pv, maximum=100, length=340).pack(pady=2)
        pl = tk.StringVar(value='')
        tk.Label(pw, textvariable=pl, bg=C_BG, fg=C_MUTED,
                 font=('微軟正黑體', 8)).pack()

        def run():
            total = len(self._images)
            for i, p in enumerate(self._images):
                img = read_img(p)
                if img is not None:
                    c_res = apply_clahe(img, clip, tile, channel)
                    src   = c_res if bin_src == 'clahe' else img
                    sg    = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                    if method == 'tophat':
                        binary, _ = binarize_tophat(sg, sigma, kernel)
                    elif method == 'threshold':
                        binary = binarize_threshold(sg, thresh)
                    elif method == 'otsu':
                        binary = binarize_otsu(sg)
                    else:
                        binary = binarize_adaptive(sg, bsize)
                    blobs   = find_blobs(binary, min_a, max_a)
                    bin_vis = draw_binary_vis(binary, blobs)
                    save_img(out_dir / f'{p.stem}_clahe{p.suffix}',  c_res)
                    save_img(out_dir / f'{p.stem}_binary{p.suffix}', bin_vis)

                pct = (i + 1) / total * 100
                self.root.after(0, pv.set, pct)
                self.root.after(0, pl.set, f'{p.name}  ({i+1}/{total})')

            def done():
                pw.destroy()
                messagebox.showinfo('完成',
                    f'已輸出 {total} 張  (_clahe + _binary)\n→ {out_dir}')
            self.root.after(0, done)

        threading.Thread(target=run, daemon=True).start()


# ═══════════════════════════════════════
#  主程式進入點
# ═══════════════════════════════════════

def main():
    init_path = None
    if len(sys.argv) > 1:
        init_path = Path(sys.argv[1])
        if not init_path.exists():
            print(f'[錯誤] 找不到路徑：{init_path}')
            sys.exit(1)

    root = tk.Tk()
    CLAHETuner(root, init_path=init_path)
    root.mainloop()


if __name__ == '__main__':
    main()
