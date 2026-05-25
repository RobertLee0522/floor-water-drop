"""
水滴辨識系統 Water Droplet Detection System
==========================================
演算法：
  - 方法 A（原始）：固定 ROI + 全局二值化
  - 方法 B（改良）：自動定位亮帶（行均值法） + Top-Hat 形態學轉換
    Top-Hat 能去除緩慢變化的背景亮度，只留下局部突出的亮點（水滴反光）
    → 對暗背景中的小亮點效果最佳

操作鍵：
  空白鍵    : 暫停 / 繼續
  +/-       : FPS ±1
  t / T     : 二值化閾值 ±10
  m         : 切換偵測方法 A/B
  q / ESC   : 離開
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # 互動式視窗；無 GUI 環境可改 'Agg'
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import time
import argparse
from collections import deque

# ─────────── 預設參數 ───────────
DEFAULT_FPS          = 5
BINARIZE_THRESH      = 200     # 方法 A 絕對閾值
TOPHAT_SIGMA         = 1.5     # 方法 B Top-Hat 閾值 = mean + sigma * std
TOPHAT_KERNEL_SIZE   = 61      # Top-Hat 結構元素大小（須為奇數）
ROW_THRESH_RATIO     = 1.3     # 自動 ROI：行均值 > 全圖均值 × 此值
ROI_PADDING          = 30      # 自動 ROI 上下各擴展 px
MIN_BLOB_AREA        = 5
MAX_BLOB_AREA        = 3000
HIST_WINDOW          = 30
HIST_CHANGE_THRESH   = 0.25
SUPPORTED_EXT        = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ─────────── 工具 ───────────

def read_image_unicode(path: Path):
    """支援中文路徑讀取圖片"""
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


def auto_roi(gray: np.ndarray):
    """根據行均值自動定位亮帶，回傳 (y1, y2)"""
    h = gray.shape[0]
    row_mean = gray.mean(axis=1)
    thresh = row_mean.mean() * ROW_THRESH_RATIO
    bright = np.where(row_mean > thresh)[0]
    if len(bright) == 0:
        return h // 3, 2 * h // 3
    y1 = max(bright.min() - ROI_PADDING, 0)
    y2 = min(bright.max() + ROI_PADDING, h)
    return int(y1), int(y2)


def detect_method_a(roi_gray: np.ndarray, thresh: int):
    """方法 A：Gaussian Blur + 全局二值化"""
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    return binary


def detect_method_b(roi_gray: np.ndarray, sigma: float = TOPHAT_SIGMA):
    """方法 B：Top-Hat 形態學（去除背景，突顯局部亮點）"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE))
    tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, k)
    # 自適應閾值
    th = float(tophat.mean() + sigma * tophat.std())
    th = max(th, 8.0)
    _, binary = cv2.threshold(tophat, int(th), 255, cv2.THRESH_BINARY)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ke, iterations=1)
    return binary, tophat, th


def find_blobs(binary: np.ndarray):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_BLOB_AREA <= area <= MAX_BLOB_AREA:
            blobs.append({
                'cx': centroids[i][0], 'cy': centroids[i][1], 'area': area,
                'x': stats[i, cv2.CC_STAT_LEFT], 'y': stats[i, cv2.CC_STAT_TOP],
            })
    return blobs


def compute_hist(gray: np.ndarray):
    h = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    s = h.sum()
    return h / s if s > 0 else h


def chi2_dist(h1, h2):
    eps = 1e-10
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)) / 2)


def draw_overlay(frame, y1, y2, blobs, count, fps, fi, total,
                 is_wet, paused, method_name):
    vis = frame.copy()
    h, w = vis.shape[:2]
    color = (0, 60, 255) if is_wet else (0, 220, 80)
    cv2.rectangle(vis, (0, y1), (w - 1, y2), color, 2)
    for b in blobs:
        cx = int(b['cx']); cy = int(b['cy']) + y1
        r = max(3, int(np.sqrt(b['area'] / np.pi)))
        cv2.circle(vis, (cx, cy), r, (0, 200, 255), 1)
        cv2.circle(vis, (cx, cy), 2, (0, 200, 255), -1)

    def put(img, txt, pos, col=(255, 255, 255), sc=0.55, th=1):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, (0,0,0), th+2, cv2.LINE_AA)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, sc, col,   th,   cv2.LINE_AA)

    status = "WET - Dense Droplets!" if is_wet else "DRY - Normal"
    sc = (0, 60, 255) if is_wet else (60, 255, 60)
    put(vis, f"Frame {fi+1}/{total}  FPS:{fps:.0f} [+/-]",   (10, 26))
    put(vis, f"Method:{method_name} [m]  Drops:{count}",     (10, 52))
    put(vis, status, (10, 78), col=sc, sc=0.65, th=2)
    if paused:
        put(vis, "[ PAUSED ]", (w//2 - 60, 32), col=(255,255,0), sc=0.75)
    return vis


# ─────────── 直方圖視窗 ───────────

class HistWin:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 7), facecolor='#111827')
        self.fig.canvas.manager.set_window_title('Histogram Analysis')
        gs = gridspec.GridSpec(2, 3, self.fig,
                               left=0.07, right=0.97, top=0.92, bottom=0.09,
                               hspace=0.5, wspace=0.35)

        def mk(pos, title):
            ax = self.fig.add_subplot(pos)
            ax.set_facecolor('#0d1117')
            ax.set_title(title, color='#e2e8f0', fontsize=8, pad=4)
            ax.tick_params(colors='#94a3b8', labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')
            return ax

        self.ax_roi     = mk(gs[0, 0], 'ROI (auto bright band)')
        self.ax_tophat  = mk(gs[0, 1], 'Top-Hat result')
        self.ax_binary  = mk(gs[0, 2], 'Binary + blobs')
        self.ax_hist    = mk(gs[1, 0], 'ROI Histogram')
        self.ax_change  = mk(gs[1, 1], f'Histogram Change x{HIST_WINDOW}f window')
        self.ax_count   = mk(gs[1, 2], 'Droplet Count per Frame')

        self.ch_data = deque(maxlen=HIST_WINDOW)
        self.cn_data = deque(maxlen=HIST_WINDOW)
        self.prev_h  = None
        self.lc, = self.ax_change.plot([], [], color='#f87171', lw=1.5)
        self.ln, = self.ax_count.plot( [], [], color='#34d399', lw=1.5)
        self.wet_line = self.ax_change.axhline(
            HIST_CHANGE_THRESH, color='yellow', ls='--', lw=1, alpha=0.8)
        for ax, xl, yl in [(self.ax_change,'Frame','Chi2'),
                           (self.ax_count, 'Frame','Count')]:
            ax.set_xlabel(xl, color='#94a3b8', fontsize=7)
            ax.set_ylabel(yl, color='#94a3b8', fontsize=7)

        plt.show(block=False)
        self._wet_spans = []

    def update(self, roi_gray, tophat_img, binary, blobs, cur_hist, count, is_wet):
        # ROI
        self.ax_roi.cla(); self.ax_roi.set_facecolor('#0d1117')
        self.ax_roi.set_title('ROI (auto bright band)', color='#e2e8f0', fontsize=8)
        self.ax_roi.imshow(roi_gray, cmap='gray', aspect='auto'); self.ax_roi.axis('off')

        # Top-Hat
        self.ax_tophat.cla(); self.ax_tophat.set_facecolor('#0d1117')
        self.ax_tophat.set_title('Top-Hat result', color='#e2e8f0', fontsize=8)
        self.ax_tophat.imshow(tophat_img, cmap='inferno', aspect='auto')
        self.ax_tophat.axis('off')

        # Binary + blobs
        self.ax_binary.cla(); self.ax_binary.set_facecolor('#0d1117')
        self.ax_binary.set_title(f'Binary + blobs ({count})', color='#e2e8f0', fontsize=8)
        bv = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        for b in blobs:
            cx, cy = int(b['cx']), int(b['cy'])
            r = max(3, int(np.sqrt(b['area']/np.pi))+1)
            cv2.circle(bv, (cx,cy), r, (255,120,0), 1)
        self.ax_binary.imshow(bv, aspect='auto'); self.ax_binary.axis('off')

        # Histogram
        self.ax_hist.cla(); self.ax_hist.set_facecolor('#0d1117')
        self.ax_hist.set_title('ROI Histogram', color='#e2e8f0', fontsize=8)
        x = np.arange(256)
        c = '#ef4444' if is_wet else '#6366f1'
        self.ax_hist.fill_between(x, cur_hist, color=c, alpha=0.8)
        self.ax_hist.set_xlim(0,255); self.ax_hist.tick_params(colors='#94a3b8',labelsize=7)
        for sp in self.ax_hist.spines.values(): sp.set_edgecolor('#2d3748')

        # Change / Count
        d = chi2_dist(self.prev_h, cur_hist) if self.prev_h is not None else 0.0
        self.prev_h = cur_hist.copy()
        self.ch_data.append(d); self.cn_data.append(count)

        yc = list(self.ch_data)
        self.lc.set_data(range(len(yc)), yc)
        self.ax_change.set_xlim(0, max(HIST_WINDOW, len(yc)))
        self.ax_change.set_ylim(0, max(max(yc)*1.2, HIST_CHANGE_THRESH*1.5, 0.05))
        for sp in self._wet_spans: sp.remove()
        self._wet_spans = [
            self.ax_change.axvspan(i-.5,i+.5,color='red',alpha=0.2)
            for i,v in enumerate(yc) if v>HIST_CHANGE_THRESH]
        self.ax_change.tick_params(colors='#94a3b8',labelsize=7)
        for sp in self.ax_change.spines.values(): sp.set_edgecolor('#2d3748')

        yn = list(self.cn_data)
        self.ln.set_data(range(len(yn)), yn)
        self.ax_count.set_xlim(0, max(HIST_WINDOW, len(yn)))
        self.ax_count.set_ylim(0, max(max(yn)*1.2, 5))
        self.ax_count.tick_params(colors='#94a3b8',labelsize=7)
        for sp in self.ax_count.spines.values(): sp.set_edgecolor('#2d3748')

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self): plt.close(self.fig)


# ─────────── 主程式 ───────────

def main():
    parser = argparse.ArgumentParser(description='水滴辨識系統')
    parser.add_argument('folder', nargs='?', default='.',
                        help='圖片資料夾（支援中文路徑）')
    parser.add_argument('--fps',    type=float, default=DEFAULT_FPS)
    parser.add_argument('--thresh', type=int,   default=BINARIZE_THRESH,
                        help='方法A 二值化閾值')
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[錯誤] 找不到資料夾：{folder}"); sys.exit(1)

    images = collect_images(folder)
    if not images:
        print(f"[錯誤] 無圖片：{folder}"); sys.exit(1)
    print(f"[載入] {len(images)} 張圖片  資料夾: {folder}")

    fps       = args.fps
    thresh    = args.thresh
    paused    = False
    frame_idx = 0
    method    = 'B'   # 預設用改良方法
    prev_hist = None
    wet_frames = []

    hist_win = HistWin()
    cv2.namedWindow('Water Droplet Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Water Droplet Detector', 1280, 720)

    print("\n操作說明：空白=暫停  +/-=FPS  t/T=閾值  m=切換方法  q/ESC=離開\n")

    tophat_cache = np.zeros((10, 10), dtype=np.uint8)  # placeholder

    while True:
        path = images[frame_idx]
        img  = read_image_unicode(path)
        if img is None:
            frame_idx = (frame_idx + 1) % len(images); continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── ROI 自動定位 ──
        y1, y2 = auto_roi(gray)
        roi_gray = gray[y1:y2, :]

        # ── 偵測 ──
        if method == 'B':
            binary, tophat_img, _ = detect_method_b(roi_gray)
            method_name = 'TopHat'
        else:
            binary = detect_method_a(roi_gray, thresh)
            tophat_img = tophat_cache
            method_name = f'Thresh{thresh}'

        blobs = find_blobs(binary)
        count = len(blobs)

        # ── 直方圖 & 是否濕 ──
        cur_hist = compute_hist(roi_gray)
        d = chi2_dist(prev_hist, cur_hist) if prev_hist is not None else 0.0
        prev_hist = cur_hist.copy()
        is_wet = (d > HIST_CHANGE_THRESH) or (count > 30)

        if is_wet and frame_idx not in wet_frames:
            wet_frames.append(frame_idx)
            print(f"  [WET] 幀{frame_idx+1:>4d} {path.name}  drops={count}  chi2={d:.4f}")

        # ── 顯示 ──
        vis = draw_overlay(img, y1, y2, blobs, count, fps,
                           frame_idx, len(images), is_wet, paused, method_name)
        cv2.imshow('Water Droplet Detector', vis)
        hist_win.update(roi_gray, tophat_img, binary, blobs, cur_hist, count, is_wet)

        # ── 鍵盤 ──
        delay = max(1, int(1000 / fps))
        key   = cv2.waitKey(1 if paused else delay) & 0xFF
        if   key in (ord('q'), 27):  break
        elif key == ord(' '):        paused = not paused
        elif key in (ord('+'), ord('=')):  fps = min(fps+1, 60); print(f"FPS→{fps:.0f}")
        elif key == ord('-'):              fps = max(fps-1, 1);  print(f"FPS→{fps:.0f}")
        elif key == ord('t'):             thresh = min(thresh+10,250); print(f"Thresh→{thresh}")
        elif key == ord('T'):             thresh = max(thresh-10,10);  print(f"Thresh→{thresh}")
        elif key == ord('m'):             method = 'A' if method=='B' else 'B'; print(f"Method→{method}")

        if not paused:
            frame_idx = (frame_idx + 1) % len(images)

    print(f"\n分析完成。共 {len(images)} 幀，水滴密集幀 {len(wet_frames)} 幀：")
    for fi in wet_frames:
        print(f"  {fi+1:>4d}: {images[fi].name}")
    hist_win.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()