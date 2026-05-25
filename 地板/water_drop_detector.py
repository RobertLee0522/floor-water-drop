"""
水滴辨識系統 Water Droplet Detection System
==========================================
ROI 模式：
  Auto  - 自動定位亮帶（行均值法）
  Draw  - 滑鼠拖曳框選任意矩形區域（按 d 進入）
  Multi - 可同時保留多個框選區域（按 a 追加）

演算法：
  方法 B（預設）：Top-Hat 形態學 + 自適應閾值
  方法 A        ：固定絕對閾值二值化

操作鍵：
  滑鼠左鍵拖曳  : 框選 ROI（需先按 d）
  d             : 進入手動繪製模式（暫停播放，可重新框選）
  a             : 追加一個新框選區域（多區域模式）
  c             : 清除所有手動 ROI，回到自動模式
  空白鍵         : 暫停 / 繼續
  +/-           : FPS ±1
  t / T         : 二值化閾值 ±10（方法 A）
  m             : 切換偵測方法 A/B
  q / ESC       : 離開
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import argparse
from collections import deque

# ─────────── 預設參數 ───────────
DEFAULT_FPS        = 5
BINARIZE_THRESH    = 200
TOPHAT_SIGMA       = 1.5
TOPHAT_KERNEL_SIZE = 61
ROW_THRESH_RATIO   = 1.3
ROI_PADDING        = 30
MIN_BLOB_AREA      = 5
MAX_BLOB_AREA      = 3000
HIST_WINDOW        = 30
HIST_CHANGE_THRESH = 0.25
SUPPORTED_EXT      = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# ROI 框線顏色（多框用不同顏色）
ROI_COLORS = [
    (0, 220, 80),    # 綠
    (255, 165, 0),   # 橘
    (0, 200, 255),   # 藍
    (220, 0, 220),   # 紫
    (0, 255, 255),   # 青
]

WIN_NAME = 'Water Droplet Detector'


# ─────────── 工具函式 ───────────

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
    """行均值法，回傳 (x1,y1,x2,y2) 全寬"""
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
    """確保矩形在圖片範圍內且 x1<x2, y1<y2"""
    x1, x2 = sorted([max(0, min(x1, W - 1)), max(0, min(x2, W - 1))])
    y1, y2 = sorted([max(0, min(y1, H - 1)), max(0, min(y2, H - 1))])
    return x1, y1, x2, y2


# ─────────── 滑鼠框選狀態 ───────────

class DrawROI:
    """管理滑鼠拖曳框選，支援顯示比例縮放"""

    def __init__(self):
        self.drawing   = False   # 是否正在拖曳
        self.active    = False   # 是否處於框選模式
        self.start     = None    # 拖曳起點（視窗座標）
        self.current   = None    # 目前滑鼠位置（視窗座標）
        self.confirmed = None    # 確認後的 ROI（圖片原始座標 x1,y1,x2,y2）
        self.scale     = 1.0     # 視窗顯示縮放比
        self.offset    = (0, 0)  # 視窗內容偏移（letterbox 情況下）

    def set_scale(self, img_w, img_h, win_w, win_h):
        """計算顯示縮放比（保持長寬比的 fit 模式）"""
        scale = min(win_w / img_w, win_h / img_h)
        self.scale = scale
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        self.offset = ((win_w - disp_w) // 2, (win_h - disp_h) // 2)

    def win_to_img(self, wx, wy):
        """視窗像素座標 → 圖片原始座標"""
        ix = int((wx - self.offset[0]) / self.scale)
        iy = int((wy - self.offset[1]) / self.scale)
        return ix, iy

    def on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start   = (x, y)
            self.current = (x, y)
            self.confirmed = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.current = (x, y)
            img_w, img_h = param
            ix1, iy1 = self.win_to_img(*self.start)
            ix2, iy2 = self.win_to_img(*self.current)
            self.confirmed = clamp_rect(ix1, iy1, ix2, iy2, img_w, img_h)

    def draw_preview(self, vis):
        """在視窗圖上畫拖曳中的虛線框"""
        if self.drawing and self.start and self.current:
            x1, y1 = self.start
            x2, y2 = self.current
            # 虛線效果：先白後黑
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 0),       1)
        return vis


# ─────────── 繪製疊加層 ───────────

def draw_overlay(frame, rois, all_blobs, total_count, fps, fi, total,
                 is_wet, paused, method_name, draw_mode):
    vis = frame.copy()
    h, w = vis.shape[:2]

    # 各 ROI 框 + 水滴圓圈
    for idx, (rect, blobs) in enumerate(zip(rois, all_blobs)):
        x1, y1, x2, y2 = rect
        color = ROI_COLORS[idx % len(ROI_COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # 角標
        label = f"ROI {idx+1}: {len(blobs)}"
        cv2.putText(vis, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (x1 + 4, y1 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        # 水滴點
        for b in blobs:
            cx = int(b['cx']) + x1
            cy = int(b['cy']) + y1
            r  = max(3, int(np.sqrt(b['area'] / np.pi)))
            cv2.circle(vis, (cx, cy), r, (0, 200, 255), 1)
            cv2.circle(vis, (cx, cy), 2, (0, 200, 255), -1)

    # HUD 文字
    def put(img, txt, pos, col=(255, 255, 255), sc=0.55, th=1):
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    sc, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, th, cv2.LINE_AA)

    status = "WET - Dense Droplets!" if is_wet else "DRY - Normal"
    sc = (0, 60, 255) if is_wet else (60, 255, 60)

    mode_str = "[ DRAW MODE - drag to select ROI ]" if draw_mode else ""
    put(vis, f"Frame {fi+1}/{total}  FPS:{fps:.0f} [+/-]", (10, 26))
    put(vis, f"Method:{method_name} [m]  Total Drops:{total_count}  ROIs:{len(rois)}", (10, 52))
    put(vis, status, (10, 78), col=sc, sc=0.65, th=2)

    hint = "d=draw  a=add  c=clear  SPACE=pause  q=quit"
    put(vis, hint, (10, h - 12), col=(200, 200, 200), sc=0.45)

    if draw_mode:
        put(vis, mode_str, (w // 2 - 200, 32), col=(255, 200, 0), sc=0.65, th=2)
    if paused:
        put(vis, "[ PAUSED ]", (w // 2 - 60, 58), col=(255, 255, 0), sc=0.75, th=2)

    return vis


# ─────────── 直方圖視窗 ───────────

class HistWin:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 7), facecolor='#111827')
        self.fig.canvas.manager.set_window_title('Histogram Analysis')
        gs = gridspec.GridSpec(2, 3, self.fig,
                               left=0.06, right=0.97, top=0.93, bottom=0.09,
                               hspace=0.5, wspace=0.32)

        def mk(pos, title):
            ax = self.fig.add_subplot(pos)
            ax.set_facecolor('#0d1117')
            ax.set_title(title, color='#e2e8f0', fontsize=8, pad=4)
            ax.tick_params(colors='#94a3b8', labelsize=7)
            for sp in ax.spines.values(): sp.set_edgecolor('#2d3748')
            return ax

        self.ax_roi    = mk(gs[0, 0], 'ROI preview')
        self.ax_tophat = mk(gs[0, 1], 'Top-Hat result')
        self.ax_binary = mk(gs[0, 2], 'Binary + blobs')
        self.ax_hist   = mk(gs[1, 0], 'ROI Histogram')
        self.ax_change = mk(gs[1, 1], f'Histogram Change (window {HIST_WINDOW}f)')
        self.ax_count  = mk(gs[1, 2], 'Droplet Count per Frame')

        self.ch_data = deque(maxlen=HIST_WINDOW)
        self.cn_data = deque(maxlen=HIST_WINDOW)
        self.prev_h  = None
        self.lc, = self.ax_change.plot([], [], color='#f87171', lw=1.5)
        self.ln, = self.ax_count.plot( [], [], color='#34d399', lw=1.5)
        self.ax_change.axhline(HIST_CHANGE_THRESH,
                               color='yellow', ls='--', lw=1, alpha=0.8)
        for ax, xl, yl in [(self.ax_change, 'Frame', 'Chi2'),
                           (self.ax_count,  'Frame', 'Count')]:
            ax.set_xlabel(xl, color='#94a3b8', fontsize=7)
            ax.set_ylabel(yl, color='#94a3b8', fontsize=7)

        plt.show(block=False)
        self._wet_spans = []

    def update(self, roi_gray, tophat_img, binary, blobs, cur_hist, count, is_wet):
        for ax, img, title, cmap in [
            (self.ax_roi,    roi_gray,    'ROI preview',    'gray'),
            (self.ax_tophat, tophat_img,  'Top-Hat result', 'inferno'),
        ]:
            ax.cla(); ax.set_facecolor('#0d1117')
            ax.set_title(title, color='#e2e8f0', fontsize=8)
            ax.imshow(img, cmap=cmap, aspect='auto'); ax.axis('off')

        self.ax_binary.cla(); self.ax_binary.set_facecolor('#0d1117')
        self.ax_binary.set_title(f'Binary + blobs ({count})', color='#e2e8f0', fontsize=8)
        bv = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        for b in blobs:
            r = max(3, int(np.sqrt(b['area'] / np.pi)) + 1)
            cv2.circle(bv, (int(b['cx']), int(b['cy'])), r, (255, 120, 0), 1)
        self.ax_binary.imshow(bv, aspect='auto'); self.ax_binary.axis('off')

        self.ax_hist.cla(); self.ax_hist.set_facecolor('#0d1117')
        self.ax_hist.set_title('ROI Histogram', color='#e2e8f0', fontsize=8)
        c = '#ef4444' if is_wet else '#6366f1'
        self.ax_hist.fill_between(np.arange(256), cur_hist, color=c, alpha=0.8)
        self.ax_hist.set_xlim(0, 255)
        self.ax_hist.tick_params(colors='#94a3b8', labelsize=7)
        for sp in self.ax_hist.spines.values(): sp.set_edgecolor('#2d3748')

        d = chi2_dist(self.prev_h, cur_hist) if self.prev_h is not None else 0.0
        self.prev_h = cur_hist.copy()
        self.ch_data.append(d); self.cn_data.append(count)

        yc = list(self.ch_data)
        self.lc.set_data(range(len(yc)), yc)
        self.ax_change.set_xlim(0, max(HIST_WINDOW, len(yc)))
        self.ax_change.set_ylim(0, max(max(yc) * 1.2, HIST_CHANGE_THRESH * 1.5, 0.05))
        for sp in self._wet_spans: sp.remove()
        self._wet_spans = [
            self.ax_change.axvspan(i - .5, i + .5, color='red', alpha=0.2)
            for i, v in enumerate(yc) if v > HIST_CHANGE_THRESH]
        self.ax_change.tick_params(colors='#94a3b8', labelsize=7)
        for sp in self.ax_change.spines.values(): sp.set_edgecolor('#2d3748')

        yn = list(self.cn_data)
        self.ln.set_data(range(len(yn)), yn)
        self.ax_count.set_xlim(0, max(HIST_WINDOW, len(yn)))
        self.ax_count.set_ylim(0, max(max(yn) * 1.2, 5))
        self.ax_count.tick_params(colors='#94a3b8', labelsize=7)
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
    parser.add_argument('--thresh', type=int,   default=BINARIZE_THRESH)
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"[錯誤] 找不到資料夾：{folder}"); sys.exit(1)
    images = collect_images(folder)
    if not images:
        print(f"[錯誤] 無圖片：{folder}"); sys.exit(1)
    print(f"[載入] {len(images)} 張圖片  資料夾: {folder}")

    fps         = args.fps
    thresh      = args.thresh
    paused      = False
    frame_idx   = 0
    method      = 'B'
    prev_hist   = None
    wet_frames  = []

    # ROI 狀態
    # manual_rois: list of (x1,y1,x2,y2) 圖片原始座標
    manual_rois  = []   # 空 = 自動模式
    draw_mode    = False

    draw_roi = DrawROI()

    # ── 建立視窗 & 設定滑鼠回呼 ──
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    # 先讀第一張取得圖片尺寸，讓 DrawROI 知道縮放比
    first_img = read_image_unicode(images[0])
    img_h, img_w = first_img.shape[:2] if first_img is not None else (720, 1280)
    draw_roi.set_scale(img_w, img_h, 1280, 720)
    cv2.setMouseCallback(WIN_NAME, draw_roi.on_mouse, (img_w, img_h))

    hist_win = HistWin()

    print("\n操作說明：")
    print("  d         : 進入框選模式（暫停後用滑鼠拖曳畫框）")
    print("  a         : 追加框選區域（保留已有框，再加一個）")
    print("  c         : 清除所有手動 ROI，回到自動偵測")
    print("  空白鍵     : 暫停 / 繼續")
    print("  +/-       : FPS ±1")
    print("  t / T     : 方法A 閾值 ±10")
    print("  m         : 切換偵測方法 TopHat / Threshold")
    print("  q / ESC   : 離開\n")

    tophat_placeholder = np.zeros((10, 10), dtype=np.uint8)

    while True:
        path = images[frame_idx]
        img  = read_image_unicode(path)
        if img is None:
            frame_idx = (frame_idx + 1) % len(images)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape

        # ── 決定本幀的 ROI 列表 ──
        if manual_rois:
            rois = manual_rois
        else:
            rois = [auto_roi_rect(gray)]

        # ── 對每個 ROI 偵測水滴 ──
        all_blobs   = []
        tophat_show = tophat_placeholder
        binary_show = tophat_placeholder
        roi_show    = gray[:10, :10]

        for rect in rois:
            x1, y1, x2, y2 = rect
            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                all_blobs.append([])
                continue

            if method == 'B':
                binary, tophat_img = detect_method_b(roi_gray)
                tophat_show = tophat_img
                method_name = 'TopHat'
            else:
                binary = detect_method_a(roi_gray, thresh)
                tophat_show = roi_gray
                method_name = f'Thresh{thresh}'

            binary_show = binary
            roi_show    = roi_gray
            all_blobs.append(find_blobs(binary))

        total_count = sum(len(b) for b in all_blobs)

        # ── 直方圖（取第一個 ROI）──
        x1, y1, x2, y2 = rois[0]
        roi_for_hist = gray[y1:y2, x1:x2]
        cur_hist = compute_hist(roi_for_hist if roi_for_hist.size > 0 else gray)
        d = chi2_dist(prev_hist, cur_hist) if prev_hist is not None else 0.0
        prev_hist = cur_hist.copy()
        is_wet = (d > HIST_CHANGE_THRESH) or (total_count > 30)

        if is_wet and frame_idx not in wet_frames:
            wet_frames.append(frame_idx)
            print(f"  [WET] 幀{frame_idx+1:>4d} {path.name}  drops={total_count}  chi2={d:.4f}")

        # ── 繪製主視窗 ──
        vis = draw_overlay(img, rois, all_blobs, total_count, fps,
                           frame_idx, len(images), is_wet, paused,
                           method_name, draw_mode)

        # 若正在拖曳，畫預覽框（視窗縮放座標）
        # 先把 vis 縮放到視窗大小再疊加拖曳框
        win_w, win_h = 1280, 720
        scale = min(win_w / img_w, win_h / img_h)
        disp_w, disp_h = int(img_w * scale), int(img_h * scale)
        vis_small = cv2.resize(vis, (disp_w, disp_h))
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        ox = (win_w - disp_w) // 2
        oy = (win_h - disp_h) // 2
        canvas[oy:oy+disp_h, ox:ox+disp_w] = vis_small
        canvas = draw_roi.draw_preview(canvas)
        cv2.imshow(WIN_NAME, canvas)

        # ── 直方圖視窗更新 ──
        hist_win.update(roi_show, tophat_show, binary_show,
                        all_blobs[0] if all_blobs else [],
                        cur_hist, total_count, is_wet)

        # ── 處理新確認的 ROI ──
        if draw_roi.confirmed is not None:
            new_rect = draw_roi.confirmed
            draw_roi.confirmed = None
            if draw_mode == 'add':
                manual_rois.append(new_rect)
                print(f"[追加 ROI #{len(manual_rois)}] {new_rect}")
            else:  # 'replace'
                manual_rois = [new_rect]
                print(f"[設定 ROI] {new_rect}")
            draw_mode   = False
            draw_roi.active = False
            paused      = False

        # ── 鍵盤 ──
        delay = max(1, int(1000 / fps))
        key   = cv2.waitKey(1 if (paused or draw_roi.active) else delay) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('+'), ord('=')):
            fps = min(fps + 1, 60);  print(f"FPS → {fps:.0f}")
        elif key == ord('-'):
            fps = max(fps - 1, 1);   print(f"FPS → {fps:.0f}")
        elif key == ord('t'):
            thresh = min(thresh + 10, 250); print(f"Thresh → {thresh}")
        elif key == ord('T'):
            thresh = max(thresh - 10, 10);  print(f"Thresh → {thresh}")
        elif key == ord('m'):
            method = 'A' if method == 'B' else 'B'; print(f"Method → {method}")
        elif key == ord('d'):
            # 進入「取代框選」模式
            draw_mode = 'replace'
            draw_roi.active = True
            draw_roi.confirmed = None
            paused = True
            print("[框選模式] 拖曳滑鼠選取新 ROI（將取代現有框）")
        elif key == ord('a'):
            # 進入「追加框選」模式
            draw_mode = 'add'
            draw_roi.active = True
            draw_roi.confirmed = None
            paused = True
            print("[追加框選] 拖曳滑鼠選取新 ROI（追加到現有框）")
        elif key == ord('c'):
            manual_rois = []
            draw_mode   = False
            draw_roi.active = False
            print("[清除] 回到自動 ROI 模式")

        if not paused and not draw_roi.active:
            frame_idx = (frame_idx + 1) % len(images)

    print(f"\n分析完成。共 {len(images)} 幀，水滴密集幀 {len(wet_frames)} 幀：")
    for fi in wet_frames:
        print(f"  {fi+1:>4d}: {images[fi].name}")
    hist_win.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()