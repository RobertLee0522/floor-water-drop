#!/usr/bin/env python3
"""
即時水滴偵測 — 觸控版（Raspberry Pi 5 + Aravis GigE 相機）
=========================================================
針對 10 吋 1280×800 觸控螢幕設計，所有操作都用大按鈕，不需要鍵盤滑鼠。

頁面：
  即時   左：即時畫面（手指拖曳可框選 ROI）  右：參數面板（相機 / 偵測 / 判定 / ROI）
  報表   監測紀錄預覽：統計摘要、水滴數曲線、WET 事件清單與截圖；可匯出 HTML

監測：按「開始監測」後每張影像都會記錄；狀態由 DRY 轉成 WET 時建立一筆事件，
並保存事件期間水滴數最多的那張偵測畫面。紀錄存在 output/sessions/<開始時間>/。

偵測演算法移植自 地板/water_drop_detector.py：
  方法 A：二值化（固定門檻）   方法 B：Top-hat（自適應門檻）
參數在離開時存到 output/live_settings.json，下次開啟自動載入。

用法：
  python3 live_drop_detector.py              # 1280×800 視窗
  python3 live_drop_detector.py --fullscreen # 全螢幕（觸控螢幕使用）
"""

import argparse
import csv
import html
import json
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path

from aravis_capture_test import Aravis, GLib, open_camera, to_bgr, try_call
import cv2
import numpy as np
try:
    from PIL import Image, ImageTk     # apt: python3-pil.imagetk；貼圖比 Tk 解析 PPM 快很多
except ImportError:
    Image = ImageTk = None

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
SETTINGS_FILE = OUTPUT_DIR / 'live_settings.json'
SESSIONS_DIR = OUTPUT_DIR / 'sessions'

WIN_W, WIN_H = 1280, 800
VIEW_STRIPS = 4             # 即時畫面拆成幾條橫帶分次貼上，每次只佔主執行緒一小段，觸控不卡
TOP_H = 84
PANEL_W = 400
VIEW_W, VIEW_H = WIN_W - PANEL_W - 12, WIN_H - TOP_H - 12

# 相機參數預設值（「重設」回到這裡；白平衡為 /1024 的倍率）
CAMERA_DEFAULTS = {
    'exposure_ms': 20.0,
    'gain': 0.0,
    'fps': 10.0,
    'wb_red': 2100,
    'wb_blue': 1750,
}
# Jinghang 相機設成其他 Bayer 格式時只改標籤、資料排列不變，標錯會整張偏綠，
# 所以依型號固定用感測器實際的排列（以紅色輸送帶實拍比對確認）
PIXEL_FORMAT_BY_MODEL = {
    'JHEM600GC': 'BayerGB8',
    'JHEM131GC': 'BayerBG8',
}
PIXEL_FORMAT_FALLBACK = 'BayerGB8'
BAYER_CHOICES = ('BayerGB8', 'BayerBG8')     # 介面上可手動選的排列


def pixel_format_for(model):
    for key, fmt in PIXEL_FORMAT_BY_MODEL.items():
        if key in (model or ''):
            return fmt
    return PIXEL_FORMAT_FALLBACK

# 偵測參數預設值（與 地板/water_drop_detector.py 相同）
DETECT_DEFAULTS = {
    'method': 'B',
    'thresh': 200,          # 方法 A 門檻
    'sigma': 1.5,           # 方法 B：mean + sigma * std
    'kernel': 61,           # 方法 B：Top-hat 核大小（原圖像素）
    'min_area': 5,          # 原圖像素
    'max_area': 3000,
    'scale': 0.5,           # 偵測前縮小比例（Pi 上全幅 Top-hat 太慢）
    'wet_count': 30,        # 水滴數超過即判定 WET
    'hist_thresh': 0.25,    # 直方圖 chi2 變化超過即判定 WET
    'wet_frames': 3,        # 連續幾張符合才切換 WET / DRY（避免閃爍）
}

# 參數面板：(key, 名稱, 下限, 上限, 步進, 小數位)；步進為 None 表示乘法調整
PARAM_GROUPS = {
    '相機': [
        ('exposure_ms', '曝光 (ms)', 0.1, 100.0, None, 1),
        ('gain', '增益 (dB)', 0.0, 24.0, 0.5, 1),
        ('fps', '幀率 (fps)', 1.0, 18.5, 1.0, 1),
        ('wb_red', '白平衡 紅', 256, 4095, 50, 0),
        ('wb_blue', '白平衡 藍', 256, 4095, 50, 0),
    ],
    '偵測': [
        ('sigma', 'B：門檻 sigma', 0.5, 6.0, 0.1, 1),
        ('kernel', 'B：Top-hat 核大小', 3, 301, 2, 0),
        ('thresh', 'A：二值化門檻', 0, 255, 5, 0),
        ('min_area', '最小面積 (px)', 1, 500, 1, 0),
        ('max_area', '最大面積 (px)', 100, 50000, 100, 0),
        ('scale', '偵測縮放比例', 0.25, 1.0, 0.05, 2),
    ],
    '判定': [
        ('wet_count', 'WET：水滴數 >', 1, 500, 1, 0),
        ('hist_thresh', 'WET：直方圖變化 >', 0.01, 2.0, 0.01, 2),
        ('wet_frames', '連續幾張才切換', 1, 30, 1, 0),
    ],
}
# 參數說明：點參數名稱或按 −/＋ 時顯示在面板空白處
GROUP_HELP = {
    '相機': '調整相機拍攝設定。\n點參數名稱可看說明。\n\n建議順序：先調曝光讓水滴清楚、背景不過曝，不夠亮再加增益。',
    '偵測': '決定「怎樣的亮點算水滴」。\n點參數名稱可看說明。\n\n誤判（沒水卻有圈）→調嚴；漏抓（有水沒圈）→調鬆。看左邊畫面的黃色圓圈即時確認。',
    '判定': '決定何時從DRY切成WET。\n點參數名稱可看說明。',
}
PARAM_HELP = {
    'exposure_ms': '每張影像的感光時間。\n畫面太暗→調大\n太亮、反光過曝、水滴拖影→調小\n曝光超過1000÷幀率毫秒時，幀率會上不去。',
    'gain': '用電子訊號放大亮度。\n曝光已經加到不能再加時才調大。\n越大雜訊越多，容易把雜訊誤判成水滴，能用0就用0。',
    'fps': '每秒拍幾張。\n調大→反應較快，但Pi負擔較重\n上方「偵測ms」很大、畫面卡頓→調小',
    'wb_red': '白平衡紅色比例，只影響畫面顏色，對偵測幾乎沒影響。\n畫面偏藍綠→調大\n畫面偏紅→調小\n也可以用下方「自動白平衡」。',
    'wb_blue': '白平衡藍色比例，只影響畫面顏色，對偵測幾乎沒影響。\n畫面偏黃→調大\n畫面偏藍→調小\n也可以用下方「自動白平衡」。',
    'method': '方法B（Top-hat，建議）：找「比周圍亮」的小亮點，整體亮度變化時也穩定。\n方法A（二值化）：亮度高於固定門檻就算，光線很穩定時才適合。\n下方只列出目前方法用得到的參數。',
    'sigma': '方法B的靈敏度：亮點要比背景亮多少才算。\n數字越小越敏感。\n雜訊、紋路被圈起來→調大\n水滴沒被圈到→調小',
    'kernel': '方法B的範圍大小（原圖像素）。比它小的亮點才會被找出來。\n約設成水滴直徑的2倍。\n大水滴只圈到一部分或漏抓→調大\n反光、大片亮區被圈→調小（數字越大也越慢）',
    'thresh': '方法A的亮度門檻（0～255），比它亮的地方算水滴。\n背景也被圈起來→調大\n水滴沒被圈到→調小',
    'min_area': '亮點面積小於此值（原圖像素）就忽略。\n很多細小雜訊點被圈→調大\n小水滴被漏掉→調小',
    'max_area': '亮點面積大於此值（原圖像素）就忽略，用來排除反光、燈影等大片亮區。\n大水滴被漏掉→調大\n大片反光被圈→調小',
    'scale': '偵測前先把影像縮小的比例，1.0為原尺寸。\n上方「偵測ms」太大→調小（較快）\n小水滴縮小後消失→調大（較準、較慢）\n面積與範圍大小會自動換算，不用跟著改。',
    'wet_count': '畫面中水滴數超過此值就判定WET。\n沒水卻常跳WET→調大\n有水卻一直DRY→調小',
    'hist_thresh': '和上一張比，畫面亮度分佈的變化量超過此值也判定WET（抓整片水漬、積水這類突然變化）。\n燈光閃爍、有人走過就誤判→調大\n大片水漬出現卻沒反應→調小',
    'bayer': '感測器的紅綠藍排列方式，選錯整張畫面會偏綠或偏藍，偵測也會受影響。\n自動：依相機型號選（JHEM600GC→GB8、JHEM131GC→BG8）\n換了相機、紅色東西拍起來變綠→改選另一個\n選擇會記住，下次開啟沿用。',
    'wet_frames': '要連續幾張都符合，才在DRY/WET之間切換，避免狀態閃爍。\n狀態跳來跳去→調大\n反應太慢→調小\n延遲秒數約=張數÷幀率',
}
METHOD_ONLY = {'sigma': 'B', 'kernel': 'B', 'thresh': 'A'}     # 只有該偵測方法才用到的參數
ROW_THRESH_RATIO = 1.3
ROI_PADDING = 30
ROI_COLORS = [(0, 220, 255), (255, 120, 0), (180, 0, 255), (0, 255, 120), (255, 0, 160)]
SNAPSHOT_W = 1280           # 事件截圖寬度
FLUSH_SEC = 60              # 監測紀錄每隔幾秒寫進硬碟一次
CHART_POINTS = 2000         # 記憶體中曲線最少保留的點數（超過 2 倍就兩兩合併）
REPORT_MAX_EVENTS = 50      # 報表頁最多顯示幾個事件縮圖（全部看匯出的 HTML）

C_BG, C_PANEL, C_CARD, C_BTN = '#0f172a', '#1e293b', '#334155', '#475569'
C_TEXT, C_MUTED = '#e2e8f0', '#94a3b8'
C_ACCENT, C_GREEN, C_RED, C_AMBER, C_PURPLE = '#0ea5e9', '#16a34a', '#dc2626', '#d97706', '#6366f1'
FONT = 'Droid Sans Fallback'


# ─────────── 偵測（移植自 地板/water_drop_detector.py）───────────

def auto_roi_rect(gray):
    h, w = gray.shape
    row_mean = gray.mean(axis=1)
    bright = np.where(row_mean > row_mean.mean() * ROW_THRESH_RATIO)[0]
    if len(bright) == 0:
        return (0, h // 3, w, 2 * h // 3)
    pad = int(ROI_PADDING * w / 1536)
    return (0, max(int(bright.min()) - pad, 0), w, min(int(bright.max()) + pad, h))


def detect_method_a(roi_gray, thresh):
    src = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, binary = cv2.threshold(src, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)


def detect_method_b(roi_gray, sigma, kernel_size):
    ksize = max(3, int(kernel_size) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat = cv2.morphologyEx(roi_gray, cv2.MORPH_TOPHAT, k)
    th = max(float(tophat.mean() + sigma * tophat.std()), 8.0)
    _, binary = cv2.threshold(tophat, int(th), 255, cv2.THRESH_BINARY)
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, ke)


def find_blobs(binary, min_area, max_area):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return [(centroids[i][0], centroids[i][1], stats[i, cv2.CC_STAT_AREA])
            for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]


def gray_hist(gray):
    h = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    s = h.sum()
    return h / s if s > 0 else h


def chi2_dist(h1, h2):
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10)) / 2)


def to_ppm(img_bgr, w, h):
    """BGR 影像縮放後轉成 PPM 位元組（不碰 Tk，可在背景執行緒做）。"""
    small = cv2.cvtColor(cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return f'P6 {w} {h} 255 '.encode() + small.tobytes()


def to_photo(img_bgr, w, h):
    """BGR 影像縮放後轉成 Tk PhotoImage（用 PPM，不需要 PIL）。"""
    return tk.PhotoImage(width=w, height=h, data=to_ppm(img_bgr, w, h), format='PPM')


def fmt_duration(sec):
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


# ─────────── 相機擷取執行緒 ───────────

class Grabber(threading.Thread):
    """背景擷取，只保留最新一張原始影像。"""

    def __init__(self, camera):
        super().__init__(daemon=True)
        self.camera = camera
        self.pixel_format = camera.get_pixel_format_as_string()
        self.lock = threading.Lock()
        self.latest = None          # (data, w, h, 序號)
        self.new_frame = threading.Event()
        self.seq = 0
        self.n_bad = 0
        self.times = []
        self.running = True
        self.stream = camera.create_stream(None, None)
        if camera.is_gv_device():
            self.stream.set_property('socket-buffer', Aravis.GvStreamSocketBuffer.AUTO)
            self.stream.set_property('packet-resend', Aravis.GvStreamPacketResend.ALWAYS)
        payload = camera.get_payload()
        for _ in range(6):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    def run(self):
        self.camera.start_acquisition()
        try:
            while self.running:
                buf = self.stream.timeout_pop_buffer(500_000)
                if buf is None:
                    continue
                ok = buf.get_status() == Aravis.BufferStatus.SUCCESS
                if ok:
                    data = buf.get_data()
                    w, h = buf.get_image_width(), buf.get_image_height()
                self.stream.push_buffer(buf)
                if not ok or len(data) < w * h:
                    self.n_bad += 1
                    continue
                with self.lock:
                    self.seq += 1
                    self.latest = (data, w, h, self.seq)
                    self.times = (self.times + [time.perf_counter()])[-20:]
                self.new_frame.set()
        finally:
            self.camera.stop_acquisition()

    def fps(self):
        with self.lock:
            t = self.times
            return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def stop(self):
        self.running = False
        self.join(timeout=3)


# ─────────── 監測紀錄 ───────────

class Session:
    """一次監測：每張影像的紀錄 + WET 事件 + 截圖。

    長時間監測也不累積記憶體：每張紀錄每 FLUSH_SEC 秒追加寫進 frames.csv，記憶體只留統計值與
    降取樣後的曲線；事件截圖寫成 JPG 後就丟掉，報表要用時再從硬碟讀。
    """

    def __init__(self, params, rois):
        self.start = datetime.now()
        self.end = None
        self.dir = SESSIONS_DIR / self.start.strftime('%Y%m%d_%H%M%S')
        (self.dir / 'events').mkdir(parents=True, exist_ok=True)
        self.params = dict(params)
        self.rois = list(rois)
        self.events = []            # dict: id, start, end, max_drops, image(只有進行中的事件), file
        self._open = None
        self.loaded = False         # True：從硬碟讀回的舊紀錄
        self._init_stats()
        self._pending = []          # 還沒寫進 frames.csv 的紀錄 (經過秒數, 水滴數, wet, chi2)
        self._last_flush = time.monotonic()
        with open(self.dir / 'frames.csv', 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(['elapsed_s', 'time', 'drops', 'wet', 'hist_chi2'])

    def _init_stats(self):
        self.n_frames = self.n_wet = self.sum_drops = 0
        self.max_drops = None
        self.last_t = 0.0
        self.series = []            # 曲線用 [經過秒數, 區間內最大水滴數]，點數固定在 CHART_POINTS～2 倍之間
        self._bucket, self._bucket_fill = 1, 0

    def _track(self, t, n, wet):
        self.n_frames += 1
        self.n_wet += bool(wet)
        self.sum_drops += n
        self.max_drops = n if self.max_drops is None else max(self.max_drops, n)
        self.last_t = t
        if self._bucket_fill == 0:
            self.series.append([t, n])
        else:
            self.series[-1][1] = max(self.series[-1][1], n)
        self._bucket_fill = (self._bucket_fill + 1) % self._bucket
        if len(self.series) >= 2 * CHART_POINTS:     # 點太多：相鄰兩點合併（取最大值保留尖峰）
            s = self.series
            self.series = [[s[i][0], max(s[i][1], s[i + 1][1])] for i in range(0, len(s), 2)]
            self._bucket *= 2
            self._bucket_fill = 0

    @property
    def running(self):
        return self.end is None

    def elapsed(self):
        return ((self.end or datetime.now()) - self.start).total_seconds()

    def add(self, n, wet, chi2, vis_bgr):
        t = self.elapsed()
        self._track(t, n, wet)
        self._pending.append((t, n, wet, chi2))
        if wet and self._open is None:
            self._open = {'id': len(self.events) + 1, 'start': t, 'end': t, 'max_drops': -1,
                          'image': None, 'file': None, 'dirty': False}
            self.events.append(self._open)
        if self._open is not None:
            ev = self._open
            ev['end'] = t
            if n > ev['max_drops']:
                ev['max_drops'] = n
                h, w = vis_bgr.shape[:2]
                ev['image'] = cv2.resize(vis_bgr, (SNAPSHOT_W, int(h * SNAPSHOT_W / w)),
                                         interpolation=cv2.INTER_AREA)
                ev['dirty'] = True
            if not wet:
                self._close_event()
        if time.monotonic() - self._last_flush >= FLUSH_SEC:
            self.flush()

    def _close_event(self):
        ev, self._open = self._open, None
        self._write_snapshot(ev)
        ev['image'] = None          # 已存成 JPG，不再佔記憶體

    def _write_snapshot(self, ev):
        if ev['image'] is not None and ev.get('dirty', True):
            ev['file'] = f"events/event_{ev['id']:03d}.jpg"
            cv2.imwrite(str(self.dir / ev['file']), ev['image'], [cv2.IMWRITE_JPEG_QUALITY, 88])
            ev['dirty'] = False

    def flush(self):
        """把累積的紀錄追加寫進 frames.csv，並更新 session.json；程式中斷時最多只少最後一段。"""
        self._last_flush = time.monotonic()
        if self.loaded:
            return
        if self._pending:
            t0 = self.start.timestamp()
            with open(self.dir / 'frames.csv', 'a', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                for t, n, wet, d in self._pending:
                    ts = datetime.fromtimestamp(t0 + t).strftime('%H:%M:%S.%f')[:-3]
                    w.writerow([f'{t:.3f}', ts, n, int(wet), f'{d:.4f}'])
            self._pending = []
        if self._open is not None:
            self._write_snapshot(self._open)     # 進行中的事件先存一張目前的截圖
        self._write_meta()

    def _write_meta(self):
        meta = {'start': self.start.isoformat(), 'end': self.end.isoformat() if self.end else None,
                'params': self.params, 'rois': [list(r) for r in self.rois],
                'events': [{k: ev[k] for k in ('id', 'start', 'end', 'max_drops', 'file')} for ev in self.events]}
        (self.dir / 'session.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    def stop(self):
        if self._open is not None:
            self._close_event()
        self.end = datetime.now()
        self.flush()

    def event_image(self, ev):
        """事件截圖：進行中的事件在記憶體，其他從 JPG 以 1/2 解碼讀回（不快取，報表縮圖夠用）。"""
        if ev.get('image') is not None:
            return ev['image']
        if ev['file'] and (self.dir / ev['file']).exists():
            return cv2.imread(str(self.dir / ev['file']), cv2.IMREAD_REDUCED_COLOR_2)
        return None

    def summary(self):
        dur = self.elapsed()
        n = self.n_frames
        wet_time = sum(e['end'] - e['start'] for e in self.events)
        return {
            '開始時間': self.start.strftime('%Y-%m-%d %H:%M:%S'),
            '結束時間': self.end.strftime('%Y-%m-%d %H:%M:%S') if self.end else '監測中',
            '監測時間': fmt_duration(dur),
            '處理影像數': f'{n}',
            '平均處理速度': f'{n / dur:.1f} fps' if dur > 0 else '-',
            'WET 事件數': f'{len(self.events)}',
            'WET 影像比例': f'{self.n_wet / n:.1%}' if n else '-',
            'WET 累計時間': fmt_duration(wet_time),
            '最大水滴數': f'{self.max_drops}' if n else '-',
            '平均水滴數': f'{self.sum_drops / n:.1f}' if n else '-',
        }

    def export(self):
        """寫出 frames.csv、session.json 與 report.html，回傳 report.html 路徑。"""
        if self.loaded:
            self._write_meta()
        else:
            self.flush()
        path = self.dir / 'report.html'
        path.write_text(render_html(self), encoding='utf-8')
        return path

    @classmethod
    def load(cls, folder):
        """從 output/sessions/<開始時間>/ 讀回一次監測（報表頁看舊紀錄用）。紀錄邊讀邊統計，不整份留在記憶體。"""
        s = cls.__new__(cls)
        s.dir, s._open, s.loaded, s._pending = folder, None, True, []
        s._init_stats()
        try:
            meta = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            meta = None
        legacy, first_time = [], None       # 舊版沒有 session.json，事件要從紀錄重建
        with open(folder / 'frames.csv', newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                first_time = first_time or row['time']
                rec = (float(row['elapsed_s']), int(row['drops']), row['wet'] == '1', float(row['hist_chi2']))
                s._track(*rec[:3])
                if meta is None:
                    legacy.append(rec)
        if meta:
            s.start = datetime.fromisoformat(meta['start'])
            s.end = datetime.fromisoformat(meta['end']) if meta.get('end') else None
            s.params, s.rois = meta['params'], [tuple(r) for r in meta['rois']]
            s.events = [dict(ev, image=None) for ev in meta['events']]
        else:
            # 時間由資料夾名稱 + 第一筆紀錄推回，事件由 wet 欄位重建
            s.start = _legacy_start(folder.name, first_time, legacy)
            s.end = None
            s.params, s.rois = _legacy_params(folder / 'report.html')
            s.events = _rebuild_events(legacy, folder)
        if s.end is None:           # 監測中途中斷：以最後一筆紀錄當結束時間
            s.end = datetime.fromtimestamp(s.start.timestamp() + s.last_t)
        return s


def _legacy_start(name, first_time, records):
    start = datetime.strptime(name, '%Y%m%d_%H%M%S')
    if not first_time:
        return start
    t = datetime.combine(start.date(), datetime.strptime(first_time, '%H:%M:%S.%f').time())
    if (start - t).total_seconds() > 3600:      # 第一筆紀錄跨過午夜
        t += timedelta(days=1)
    return datetime.fromtimestamp(t.timestamp() - records[0][0])


def _legacy_params(report_file):
    """從舊版 report.html 的「偵測參數」表格讀回參數。"""
    labels = {label: (key, digits) for group in PARAM_GROUPS.values() for key, label, *_, digits in group}
    params, rois = {}, []
    try:
        text = report_file.read_text(encoding='utf-8')
    except OSError:
        return params, rois
    for k, v in re.findall(r'<tr><th>(.*?)</th><td>(.*?)</td></tr>', text):
        k, v = html.unescape(k), html.unescape(v)
        if k == '偵測方法':
            params['method'] = 'B' if v.startswith('B') else 'A'
        elif k == 'ROI':
            rois = [tuple(map(int, m)) for m in re.findall(r'\((\d+), (\d+), (\d+), (\d+)\)', v)]
        elif k in labels:
            key, digits = labels[k]
            try:
                params[key] = int(float(v)) if digits == 0 else float(v)
            except ValueError:
                pass
    return params, rois


def _rebuild_events(records, folder):
    """依 Session.add 的規則，從每張影像的 wet 欄位重建 WET 事件。"""
    events, ev = [], None
    for t, n, wet, _ in records:
        if wet and ev is None:
            ev = {'id': len(events) + 1, 'start': t, 'end': t, 'max_drops': -1, 'image': None, 'file': None}
            events.append(ev)
        if ev is not None:
            ev['end'], ev['max_drops'] = t, max(ev['max_drops'], n)
            if not wet:
                ev = None
    for ev in events:
        f = f"events/event_{ev['id']:03d}.jpg"
        if (folder / f).exists():
            ev['file'] = f
    return events


# ─────────── HTML 報表 ───────────

def chart_svg(session, width=1000, height=280):
    recs = session.series
    pad_l, pad_r, pad_t, pad_b = 48, 16, 14, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    dur = max(session.elapsed(), 1e-6)
    thresh = session.params.get('wet_count', DETECT_DEFAULTS['wet_count'])
    ymax = max([r[1] for r in recs] + [thresh]) * 1.1 or 1
    x = lambda t: pad_l + t / dur * pw
    y = lambda v: pad_t + ph - v / ymax * ph
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="水滴數時間曲線">']
    for ev in session.events:
        x0, x1 = x(ev['start']), x(ev['end'])
        parts.append(f'<rect class="wet" x="{x0:.1f}" y="{pad_t}" width="{max(x1 - x0, 2):.1f}" height="{ph}"/>')
    for i in range(5):
        v = ymax * i / 4
        parts.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                     f'<text class="ax" x="{pad_l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>')
    for i in range(6):
        t = dur * i / 5
        parts.append(f'<text class="ax" x="{x(t):.1f}" y="{height - 10}" text-anchor="middle">{fmt_duration(t)}</text>')
    parts.append(f'<line class="th" x1="{pad_l}" x2="{width - pad_r}" y1="{y(thresh):.1f}" y2="{y(thresh):.1f}"/>')
    if recs:
        # 點太多時每個區間取最大值，避免檔案過大又保留尖峰
        bucket = max(1, len(recs) // 1500)
        pts = []
        for i in range(0, len(recs), bucket):
            chunk = recs[i:i + bucket]
            pts.append(f'{x(chunk[0][0]):.1f},{y(max(r[1] for r in chunk)):.1f}')
        parts.append(f'<polyline class="line" points="{" ".join(pts)}"/>')
    parts.append('</svg>')
    return ''.join(parts)


def param_lines(params, rois):
    lines = [('偵測方法', 'B：Top-hat' if params.get('method') == 'B' else 'A：二值化')]
    for group in PARAM_GROUPS.values():
        lines += [(label, params.get(key, '—')) for key, label, *_ in group]
    lines.append(('ROI', ', '.join(str(tuple(r)) for r in rois) or '自動'))
    return lines


def render_html(session):
    e = html.escape
    summary = session.summary()
    p = session.params
    ev_rows = []
    for ev in session.events:
        t0 = datetime.fromtimestamp(session.start.timestamp() + ev['start']).strftime('%H:%M:%S')
        img = (f'<a href="{e(ev["file"])}"><img src="{e(ev["file"])}" alt="事件 {ev["id"]} 截圖" loading="lazy"></a>'
               if ev['file'] else '—')
        ev_rows.append(f'<tr><td>{ev["id"]}</td><td>{t0}</td><td>{fmt_duration(ev["end"] - ev["start"])}</td>'
                       f'<td>{ev["max_drops"]}</td><td>{img}</td></tr>')
    return f'''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>水滴偵測報表 {e(summary["開始時間"])}</title>
<style>
:root {{ --bg:#f8fafc; --card:#fff; --text:#0f172a; --muted:#64748b; --line:#e2e8f0;
        --accent:#0284c7; --wet:rgba(239,68,68,.16); --th:#ef4444; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --line:#334155;
          --accent:#38bdf8; --wet:rgba(239,68,68,.25); --th:#f87171; }} }}
body {{ margin:0; background:var(--bg); color:var(--text);
       font-family:"Noto Sans TC","Microsoft JhengHei","Droid Sans Fallback",sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }} h2 {{ font-size:1.1rem; margin:28px 0 10px; }}
.sub {{ color:var(--muted); margin:0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:10px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
.card .k {{ color:var(--muted); font-size:.8rem; }} .card .v {{ font-size:1.2rem; font-weight:600; }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; overflow-x:auto; }}
svg {{ width:100%; height:auto; display:block; }}
svg .grid {{ stroke:var(--line); }} svg .ax {{ fill:var(--muted); font-size:11px; }}
svg .line {{ fill:none; stroke:var(--accent); stroke-width:1.5; }}
svg .th {{ stroke:var(--th); stroke-dasharray:5 4; }} svg .wet {{ fill:var(--wet); }}
.legend {{ color:var(--muted); font-size:.85rem; margin-top:6px; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ text-align:left; padding:8px; border-bottom:1px solid var(--line); vertical-align:middle; }}
th {{ color:var(--muted); font-weight:500; font-size:.85rem; }}
td img {{ width:240px; max-width:100%; border-radius:6px; display:block; }}
</style></head><body><main>
<h1>水滴偵測報表</h1>
<p class="sub">{e(summary["開始時間"])} → {e(summary["結束時間"])}　·　產生於 {datetime.now():%Y-%m-%d %H:%M:%S}</p>
<h2>統計摘要</h2>
<div class="cards">{"".join(f'<div class="card"><div class="k">{e(k)}</div><div class="v">{e(v)}</div></div>' for k, v in summary.items())}</div>
<h2>水滴數時間曲線</h2>
<div class="panel">{chart_svg(session)}
<div class="legend">藍線：水滴數　紅色虛線：WET 門檻（{e(str(p.get("wet_count")))}）　紅色底色：WET 事件期間</div></div>
<h2>WET 事件（{len(session.events)}）</h2>
<div class="panel"><table><thead><tr><th>#</th><th>開始</th><th>持續</th><th>最大水滴數</th><th>截圖（點開看大圖）</th></tr></thead>
<tbody>{"".join(ev_rows) or '<tr><td colspan="5">監測期間沒有 WET 事件</td></tr>'}</tbody></table></div>
<h2>偵測參數</h2>
<div class="panel"><table><tbody>{"".join(f"<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>" for k, v in param_lines(p, session.rois))}</tbody></table></div>
<p class="sub">每張影像的原始紀錄：frames.csv</p>
</main></body></html>
'''


# ─────────── 主程式 ───────────

class LiveDropApp:
    def __init__(self, root, camera, fullscreen=False):
        self.root = root
        self.camera = camera
        self.grabber = Grabber(camera)
        self.defaults = {**CAMERA_DEFAULTS, **DETECT_DEFAULTS}
        self.params = dict(self.defaults)
        self.specs = {spec[0]: spec for group in PARAM_GROUPS.values() for spec in group}
        self.history = []           # 復原用：每次變更前的參數快照
        self._commit_job = None
        self._shown_method = None   # 偵測分頁目前列出哪個方法的參數
        self._help_job = None       # 畫面上說明框的自動關閉計時

        self.model = try_call(camera.get_model_name, default='') or ''
        self.bayer = 'auto'         # 顏色排列：'auto' 或 BAYER_CHOICES 之一
        self.rois = []              # 手動 ROI（原圖座標）
        self._drag = None
        self.paused = False
        self.prev_hist = None
        self.last_seq = 0
        self.frame_bgr = None
        self.vis_bgr = None
        self.n_drops = 0
        self.chi2 = 0.0
        self.is_wet = False
        self._streak = 0            # 與目前狀態相反的連續張數
        self.proc_ms = 0.0
        self.view_scale = 1.0
        self.view_offset = (0, 0)
        self.view_size = (VIEW_W, VIEW_H)   # 畫布目前大小（由 <Configure> 更新，背景執行緒讀）
        self._photos = []           # 即時畫面各橫帶的 PhotoImage
        self._view_key = None       # (寬, 高, 位移)：改變時才重建 PhotoImage
        self._strip_job = None
        self._pending_strips = None
        self.session = None
        self.page = 'live'
        self.group = '相機'
        self.report_dir = None      # 報表頁正在看的紀錄資料夾（None：最新一筆）
        self._loaded = {}           # 已讀回的舊紀錄快取

        # 偵測在背景執行緒做，主執行緒只貼圖，觸控操作才不會卡
        self.lock = threading.Lock()    # 保護 session（背景寫紀錄 vs 主執行緒停止 / 匯出 / 畫報表）
        self.result = None          # (原圖, 偵測畫面, 水滴數, chi2, 顯示用影像, 序號)
        self.running = True
        self.worker = threading.Thread(target=self._process_loop, daemon=True)

        self._build_ui()
        self._load_settings()
        if fullscreen:
            self.toggle_fullscreen()
        self.grabber.start()
        self.worker.start()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<Configure>', self._on_root_resize)
        self.root.after(30, self._tick)

    # ── 相機 ──
    def _set_wb(self, channel, value):
        self.camera.set_string('BalanceRatioSelector', channel)
        self.camera.set_integer('BalanceRatio', int(value))

    def _apply_camera(self, key, value):
        cam = self.camera
        try:
            if key == 'exposure_ms':
                try_call(cam.set_exposure_time_auto, Aravis.Auto.OFF)
                cam.set_exposure_time(float(value) * 1000)
            elif key == 'gain':
                try_call(cam.set_gain_auto, Aravis.Auto.OFF)
                cam.set_gain(float(value))
            elif key == 'fps':
                cam.set_frame_rate(float(value))
            elif key == 'wb_red':
                self._set_wb('Red', value)
            elif key == 'wb_blue':
                self._set_wb('Blue', value)
        except GLib.Error as e:
            print(f'[警告] 設定 {key}={value} 失敗：{e.message}')

    def set_bayer(self, choice):
        # 相機改格式只換標籤、資料排列不變，所以只要換解拜耳方式，不必停止擷取
        self.bayer = choice
        self.grabber.pixel_format = pixel_format_for(self.model) if choice == 'auto' else choice
        self._refresh_bayer()
        self._save_bayer()

    def _refresh_bayer(self):
        if self.group != '相機':
            return
        for v, b in self.bayer_btns.items():
            c = C_ACCENT if self.bayer == v else C_BTN
            b.config(bg=c, activebackground=c)
        auto = pixel_format_for(self.model).replace('Bayer', '')
        self.bayer_btns['auto'].config(text=f'自動 {auto}')

    # ── UI 元件 ──
    def _btn(self, parent, text, cmd, bg=C_CARD, size=16, repeat=False, **kw):
        opts = dict(text=text, command=cmd, bg=bg, fg='white', activebackground=bg,
                    activeforeground='white', relief='flat', bd=0, highlightthickness=0,
                    font=(FONT, size, 'bold'))
        if repeat:      # 按住連續調整
            opts.update(repeatdelay=400, repeatinterval=90)
        opts.update(kw)
        return tk.Button(parent, **opts)

    def _build_ui(self):
        r = self.root
        r.title('即時水滴偵測')
        r.configure(bg=C_BG)
        r.geometry(f'{WIN_W}x{WIN_H}')

        # 上方狀態列：狀態、水滴數、主要動作
        top = tk.Frame(r, bg=C_PANEL, height=TOP_H)
        top.pack(fill='x')
        top.pack_propagate(False)
        self.status_lbl = tk.Label(top, text='DRY', bg=C_GREEN, fg='white', width=5,
                                   font=(FONT, 28, 'bold'))
        self.status_lbl.pack(side='left', fill='y', padx=(0, 12))
        self.count_lbl = tk.Label(top, text='0', bg=C_PANEL, fg=C_TEXT, font=(FONT, 32, 'bold'), width=4,
                                  anchor='e')
        self.count_lbl.pack(side='left')
        tk.Label(top, text='顆', bg=C_PANEL, fg=C_MUTED, font=(FONT, 16)).pack(side='left', padx=(2, 12))
        self.info_lbl = tk.Label(top, text='', bg=C_PANEL, fg=C_MUTED, font=(FONT, 12), justify='left',
                                 anchor='w')
        self.info_lbl.pack(side='left', fill='x', expand=True)

        self._btn(top, '離開', self.close, bg=C_BTN, width=4).pack(side='right', fill='y', padx=(4, 8), pady=8)
        self.fs_btn = self._btn(top, '全螢幕', self.toggle_fullscreen, bg=C_BTN, width=5)
        self.fs_btn.pack(side='right', fill='y', padx=4, pady=8)
        self.page_btn = self._btn(top, '報表', self.toggle_page, bg=C_PURPLE, size=18, width=5)
        self.page_btn.pack(side='right', fill='y', padx=4, pady=8)
        self.mon_btn = self._btn(top, '開始監測', self.toggle_monitor, bg=C_GREEN, size=18, width=7)
        self.mon_btn.pack(side='right', fill='y', padx=4, pady=8)

        self.body = tk.Frame(r, bg=C_BG)
        self.body.pack(fill='both', expand=True)
        self._build_live_page()
        self._build_report_page()
        self.live_page.pack(fill='both', expand=True)

    def _build_live_page(self):
        page = self.live_page = tk.Frame(self.body, bg=C_BG)
        # 右側面板先 pack 固定寬度，畫布再填滿剩下空間（全螢幕時跟著放大）
        panel = tk.Frame(page, bg=C_PANEL, width=PANEL_W)
        panel.pack(side='right', fill='y', pady=6, padx=(0, 6))
        panel.pack_propagate(False)

        self.canvas = tk.Canvas(page, width=VIEW_W, height=VIEW_H, bg='black', highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True, padx=6, pady=6)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Configure>', self._on_canvas_resize)

        # 分頁按鈕
        tabs = tk.Frame(panel, bg=C_PANEL)
        tabs.pack(fill='x', padx=6, pady=(6, 6))
        self.tab_btns = {}
        for name in list(PARAM_GROUPS) + ['ROI']:
            b = self._btn(tabs, name, lambda n=name: self._show_group(n), size=16)
            b.pack(side='left', expand=True, fill='x', padx=2, ipady=12)
            self.tab_btns[name] = b

        bottom = tk.Frame(panel, bg=C_PANEL)
        bottom.pack(fill='x', side='bottom', padx=6, pady=6)
        for text, cmd, color in (('復原', self.undo, C_ACCENT), ('重設', self.reset, C_AMBER),
                                 ('暫停', self.toggle_pause, C_PURPLE), ('存圖', self.save, C_GREEN)):
            b = self._btn(bottom, text, cmd, bg=color, size=17)
            b.pack(side='left', expand=True, fill='x', padx=2, ipady=14)
            if text == '暫停':
                self.pause_btn = b

        self.group_frame = tk.Frame(panel, bg=C_PANEL)
        self.group_frame.pack(fill='both', expand=True, padx=6)
        self.value_lbls = {}
        self._show_group(self.group)

    def _show_group(self, name):
        self.group = name
        for n, b in self.tab_btns.items():
            c = C_ACCENT if n == name else C_CARD
            b.config(bg=c, activebackground=c)
        for w in self.group_frame.winfo_children():
            w.destroy()
        self.value_lbls = {}
        f = self.group_frame
        if name == 'ROI':
            tk.Label(f, text='在左邊畫面上用手指拖曳\n就能框選偵測範圍（可框多個）。\n\n沒有框選時，自動找亮帶當 ROI。',
                     bg=C_PANEL, fg=C_TEXT, font=(FONT, 15), justify='left').pack(anchor='w', pady=12)
            self.roi_lbl = tk.Label(f, text='', bg=C_PANEL, fg=C_ACCENT, font=(FONT, 17, 'bold'))
            self.roi_lbl.pack(anchor='w', pady=8)
            self._btn(f, '刪除最後一個 ROI', self.pop_roi).pack(fill='x', pady=5, ipady=16)
            self._btn(f, '清除全部（改回自動）', self.clear_roi).pack(fill='x', pady=5, ipady=16)
            self._update_roi_label()
            return
        if name == '偵測':
            row = tk.Frame(f, bg=C_PANEL)
            row.pack(fill='x', pady=(0, 4))
            self.method_btns = {}
            for val, text in (('B', '方法 B  Top-hat'), ('A', '方法 A  二值化')):
                b = self._btn(row, text, lambda v=val: (self._set_param('method', v), self._show_help('method')),
                              size=14)
                b.pack(side='left', expand=True, fill='x', padx=2, ipady=10)
                self.method_btns[val] = b
            self._shown_method = self.params['method']
            self._refresh_method()
        for key, label, lo, hi, step, digits in PARAM_GROUPS[name]:
            if METHOD_ONLY.get(key, self.params['method']) != self.params['method']:
                continue
            row = tk.Frame(f, bg=C_CARD)
            row.pack(fill='x', pady=3)
            self._btn(row, '＋', lambda k=key: self._step(k, +1), bg=C_BTN, size=24, width=3,
                      repeat=True).pack(side='right', fill='y', padx=(3, 4), pady=4)
            self._btn(row, '－', lambda k=key: self._step(k, -1), bg=C_BTN, size=24, width=3,
                      repeat=True).pack(side='right', fill='y', padx=3, pady=4)
            left = tk.Frame(row, bg=C_CARD)
            left.pack(side='left', fill='x', expand=True, padx=10, pady=2)
            tk.Label(left, text=label, bg=C_CARD, fg=C_MUTED, font=(FONT, 12), anchor='w').pack(fill='x')
            v = tk.Label(left, text='', bg=C_CARD, fg=C_TEXT, font=(FONT, 20, 'bold'), anchor='w')
            v.pack(fill='x')
            self.value_lbls[key] = v
            for w in (row, left, *left.winfo_children()):      # 點參數名稱看說明
                w.bind('<ButtonPress-1>', lambda e, k=key: self._show_help(k))
        if name == '相機':
            row = tk.Frame(f, bg=C_CARD)
            row.pack(fill='x', pady=3)
            lbl = tk.Label(row, text='顏色\n排列', bg=C_CARD, fg=C_MUTED, font=(FONT, 12))
            lbl.pack(side='left', padx=(10, 6))
            lbl.bind('<ButtonPress-1>', lambda e: self._show_help('bayer'))
            self.bayer_btns = {}
            for val in ('auto', *BAYER_CHOICES):
                b = self._btn(row, val.replace('Bayer', ''),
                              lambda v=val: (self.set_bayer(v), self._show_help('bayer')), bg=C_BTN, size=14)
                b.pack(side='left', expand=True, fill='x', padx=2, pady=4, ipady=6)
                self.bayer_btns[val] = b
            self._refresh_bayer()
            self._btn(f, '自動白平衡（畫面中央放白紙）', self.auto_wb, size=14).pack(fill='x', pady=6, ipady=10)
        # 說明區：填滿面板剩下的空白
        self.help_lbl = tk.Label(f, text='', bg=C_PANEL, fg=C_TEXT, font=(FONT, 13), justify='left', anchor='nw',
                                 wraplength=PANEL_W - 40)
        self.help_lbl.pack(fill='both', expand=True, pady=(10, 0))
        self._show_help(None)
        self._refresh_values()

    def _show_help(self, key):
        if key is None:
            text = GROUP_HELP[self.group]
        else:
            name = {'method': '偵測方法', 'bayer': '顏色排列'}.get(key) or self.specs[key][1]
            text = f'【{name}】\n{PARAM_HELP[key]}'
        self.hide_help_overlay()
        self.help_lbl.config(text=text)
        self.root.update_idletasks()
        if key is not None and self.help_lbl.winfo_reqheight() > self.help_lbl.winfo_height():
            # 面板空白不夠（例如 1280×800）：改浮在左邊畫面上
            self.help_lbl.config(text=f'【{name}】說明顯示在左邊畫面')
            self._show_help_overlay(text)

    def _show_help_overlay(self, text):
        c = self.canvas
        t = c.create_text(28, 28, text=text + '\n\n（點此關閉）', anchor='nw', fill=C_TEXT,
                          font=(FONT, 16), width=min(520, c.winfo_width() - 60), tags='help')
        x1, y1, x2, y2 = c.bbox(t)
        c.create_rectangle(x1 - 16, y1 - 16, x2 + 16, y2 + 16, fill=C_PANEL, outline=C_ACCENT, width=2,
                           tags='help')
        c.tag_raise(t)
        self._help_job = self.root.after(20000, self.hide_help_overlay)

    def hide_help_overlay(self):
        self.canvas.delete('help')
        if self._help_job is not None:
            self.root.after_cancel(self._help_job)
            self._help_job = None

    def _refresh_values(self):
        for key, lbl in self.value_lbls.items():
            lbl.config(text=f'{self.params[key]:.{self.specs[key][5]}f}')

    def _refresh_method(self):
        if self.group == '偵測' and self._shown_method != self.params['method']:
            self._show_group('偵測')        # 方法改變：換成新方法的參數列
            self._show_help('method')
            return
        if self.group == '偵測':
            for v, b in self.method_btns.items():
                c = C_ACCENT if self.params['method'] == v else C_BTN
                b.config(bg=c, activebackground=c)

    # ── 參數 / 復原 ──
    def _step(self, key, direction):
        _, _, lo, hi, step, digits = self.specs[key]
        v = self.params[key]
        if step is None:            # 曝光：乘法調整，短曝光也能細調
            v = v * 1.12 if direction > 0 else v / 1.12
        else:
            v = v + direction * step
        v = round(min(max(v, lo), hi), digits)
        self._set_param(key, int(v) if digits == 0 else v)
        self._show_help(key)

    def _set_param(self, key, value):
        if self.params[key] == value:
            return
        # 連續按（或按住）時合併成一筆復原紀錄
        if self._commit_job is None:
            self.history = (self.history + [dict(self.params)])[-50:]
        else:
            self.root.after_cancel(self._commit_job)
        self._commit_job = self.root.after(800, self._end_commit)
        self.params[key] = value
        if key in CAMERA_DEFAULTS:
            self._apply_camera(key, value)
        self._after_params_change()

    def _end_commit(self):
        self._commit_job = None

    def _apply_all(self, params, force=False):
        old, self.params = self.params, dict(params)
        for k in CAMERA_DEFAULTS:
            if force or old.get(k) != self.params[k]:
                self._apply_camera(k, self.params[k])
        self._after_params_change()

    def _after_params_change(self):
        self._refresh_values()
        self._refresh_method()
        if self.session and self.session.running:
            self.session.params = dict(self.params)

    def undo(self):
        if self._commit_job is not None:
            self.root.after_cancel(self._commit_job)
            self._commit_job = None
        if self.history:
            self._apply_all(self.history.pop())

    def reset(self):
        self.history = (self.history + [dict(self.params)])[-50:]
        self._apply_all(self.defaults)

    def _load_settings(self):
        params = dict(self.defaults)
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
            params.update({k: v for k, v in saved.get('params', {}).items() if k in params})
            self.rois = [tuple(r) for r in saved.get('rois', [])]
            if saved.get('bayer') in BAYER_CHOICES:
                self.set_bayer(saved['bayer'])
        except (OSError, ValueError):
            pass
        self._apply_all(params, force=True)
        self._update_roi_label()

    def _save_settings(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps({'params': self.params, 'rois': self.rois, 'bayer': self.bayer},
                                            ensure_ascii=False, indent=2))

    def _save_bayer(self):
        # 顏色排列選了就記住（不必等按「存檔」），其他參數維持檔案裡原本的值
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
        except (OSError, ValueError):
            saved = {}
        saved['bayer'] = self.bayer
        OUTPUT_DIR.mkdir(exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(saved, ensure_ascii=False, indent=2))

    # ── 白平衡 ──
    def auto_wb(self):
        if self.frame_bgr is None:
            return
        h, w = self.frame_bgr.shape[:2]
        box = self.frame_bgr[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)]
        b, g, r = (float(box[..., i].mean()) for i in range(3))
        if min(b, g, r) < 5 or max(b, g, r) > 250:
            self.info_lbl.config(text=f'白平衡失敗：中央太暗或過曝（B={b:.0f} G={g:.0f} R={r:.0f}）')
            return
        new = dict(self.params)
        new['wb_red'] = int(min(max(round(new['wb_red'] * g / r), 256), 4095))
        new['wb_blue'] = int(min(max(round(new['wb_blue'] * g / b), 256), 4095))
        self.history = (self.history + [dict(self.params)])[-50:]
        self._apply_all(new)

    # ── ROI ──
    def _canvas_to_img(self, x, y):
        s = self.view_scale
        return int((x - self.view_offset[0]) / s), int((y - self.view_offset[1]) / s)

    def _on_press(self, e):
        if 'help' in self.canvas.gettags('current'):
            self.hide_help_overlay()
            self._drag = None
            return
        self._drag = (e.x, e.y)

    def _on_drag(self, e):
        if self._drag:
            self.canvas.delete('drag')
            self.canvas.create_rectangle(*self._drag, e.x, e.y, outline='#facc15', width=3, tags='drag')

    def _on_release(self, e):
        self.canvas.delete('drag')
        if not self._drag or self.frame_bgr is None:
            return
        (x1, y1), (x2, y2) = self._canvas_to_img(*self._drag), self._canvas_to_img(e.x, e.y)
        self._drag = None
        h, w = self.frame_bgr.shape[:2]
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
        if x2 - x1 > 80 and y2 - y1 > 80:   # 太小的視為誤觸
            self.rois.append((x1, y1, x2, y2))
            self.prev_hist = None
            self._update_roi_label()

    def pop_roi(self):
        if self.rois:
            self.rois.pop()
            self.prev_hist = None
            self._update_roi_label()

    def clear_roi(self):
        self.rois = []
        self.prev_hist = None
        self._update_roi_label()

    def _update_roi_label(self):
        if self.group == 'ROI':
            self.roi_lbl.config(text=f'目前：手動 ROI × {len(self.rois)}' if self.rois else '目前：自動 ROI')

    # ── 監測 / 頁面 ──
    def toggle_monitor(self):
        with self.lock:
            if self.session and self.session.running:
                self.session.stop()
                self.session.export()
                started = False
            else:
                self.session = Session(self.params, self.rois)
                started = True
        if started:
            self.mon_btn.config(text='停止監測', bg=C_RED, activebackground=C_RED)
        else:
            self.mon_btn.config(text='開始監測', bg=C_GREEN, activebackground=C_GREEN)
        if self.page == 'report':
            self.report_dir = None
            self._render_report()

    def toggle_page(self):
        if self.page == 'live':
            self.page = 'report'
            self.live_page.pack_forget()
            self.report_page.pack(fill='both', expand=True)
            self.page_btn.config(text='即時')
            self.report_dir = None
            self._render_report()
        else:
            self.page = 'live'
            self.report_page.pack_forget()
            self.live_page.pack(fill='both', expand=True)
            self.page_btn.config(text='報表')

    def toggle_fullscreen(self):
        fs = not bool(self.root.attributes('-fullscreen'))
        self.root.attributes('-fullscreen', fs)
        self.fs_btn.config(text='視窗' if fs else '全螢幕')
        if self.page == 'report':  # 等視窗尺寸更新後再依新寬度重畫報表
            self.root.after(200, self._render_report)

    def _on_root_resize(self, e):
        # labwc + Xwayland：視窗放大後被搬移的子元件（右側面板按鈕）內容會變黑，又不會收到 Expose，
        # 要等別的事件才重畫。尺寸穩定後對所有元件送一次 Expose 強制重繪。
        if e.widget is not self.root or (e.width, e.height) == getattr(self, '_root_size', None):
            return
        self._root_size = (e.width, e.height)
        if getattr(self, '_expose_job', None):
            self.root.after_cancel(self._expose_job)
        self._expose_job = self.root.after(150, self._force_redraw)

    def _force_redraw(self):
        self._expose_job = None
        stack = [self.root]
        while stack:
            w = stack.pop()
            stack.extend(w.winfo_children())
            if w is not self.root and w.winfo_ismapped():
                w.event_generate('<Expose>')

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text='繼續' if self.paused else '暫停')

    # ── 報表頁 ──
    def _build_report_page(self):
        page = self.report_page = tk.Frame(self.body, bg=C_BG)
        bar = tk.Frame(page, bg=C_BG)
        bar.pack(fill='x', padx=8, pady=(8, 6))
        self.older_btn = self._btn(bar, '◀ 較舊', lambda: self._report_nav(+1))
        self.older_btn.pack(side='left', padx=(0, 4), ipady=12, ipadx=10)
        self.newer_btn = self._btn(bar, '較新 ▶', lambda: self._report_nav(-1))
        self.newer_btn.pack(side='left', padx=(0, 12), ipady=12, ipadx=10)
        self.report_title = tk.Label(bar, text='', bg=C_BG, fg=C_TEXT, font=(FONT, 16, 'bold'), anchor='w')
        self.report_title.pack(side='left', fill='x', expand=True)
        self._btn(bar, '重新整理', self._render_report).pack(side='right', padx=4, ipady=12, ipadx=14)
        self._btn(bar, '匯出並開啟 HTML', self.export_report, bg=C_ACCENT).pack(
            side='right', padx=4, ipady=12, ipadx=14)

        # 手指上下拖曳即可捲動
        self.rcanvas = tk.Canvas(page, bg=C_BG, highlightthickness=0)
        self.rcanvas.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        self.rcanvas.bind('<ButtonPress-1>', lambda e: self.rcanvas.scan_mark(0, e.y))
        self.rcanvas.bind('<B1-Motion>', lambda e: self.rcanvas.scan_dragto(0, e.y, gain=1))
        self.rcanvas.bind('<Button-4>', lambda e: self.rcanvas.yview_scroll(-3, 'units'))
        self.rcanvas.bind('<Button-5>', lambda e: self.rcanvas.yview_scroll(3, 'units'))
        self._report_photos = []

    def _report_dirs(self):
        """所有監測紀錄資料夾（新到舊）；這次監測就算還沒匯出也列在最前面。"""
        dirs = sorted((d for d in SESSIONS_DIR.glob('*') if (d / 'frames.csv').exists()), reverse=True)
        if self.session and self.session.dir not in dirs:
            dirs.insert(0, self.session.dir)
        return dirs

    def _report_session(self, folder):
        if self.session and folder == self.session.dir:
            return self.session
        if folder not in self._loaded:
            try:
                self._loaded[folder] = Session.load(folder)
            except (OSError, ValueError, KeyError, IndexError) as e:
                print(f'[警告] 讀取紀錄 {folder.name} 失敗：{e}')
                self._loaded[folder] = None
        return self._loaded[folder]

    def _report_nav(self, step):
        dirs = self._report_dirs()
        if self.report_dir in dirs:
            i = min(max(dirs.index(self.report_dir) + step, 0), len(dirs) - 1)
            self.report_dir = dirs[i]
            self._render_report()

    def _render_report(self):
        with self.lock:
            self._render_report_locked()

    def _render_report_locked(self):
        c = self.rcanvas
        c.delete('all')
        self._report_photos = []
        dirs = self._report_dirs()
        if self.report_dir not in dirs:
            self.report_dir = dirs[0] if dirs else None
        idx = dirs.index(self.report_dir) if self.report_dir else 0
        for b, ok in ((self.older_btn, idx < len(dirs) - 1), (self.newer_btn, idx > 0)):
            b.config(state='normal' if ok else 'disabled', bg=C_CARD if ok else C_BG)
        s = self._report_session(self.report_dir) if self.report_dir else None
        W = max(self.rcanvas.winfo_width(), WIN_W - 16)
        if s is None and self.report_dir:
            self.report_title.config(text=f'紀錄 {idx + 1}/{len(dirs)}  {self.report_dir.name} 讀取失敗')
            c.configure(scrollregion=(0, 0, W, 400))
            return
        if s is None:
            self.report_title.config(text='尚無監測紀錄')
            c.create_text(W // 2, 180, text='按上方「開始監測」開始記錄，\n之後回到這頁就能預覽報表。',
                          fill=C_MUTED, font=(FONT, 20), justify='center')
            c.configure(scrollregion=(0, 0, W, 400))
            return
        self.report_title.config(text=f'紀錄 {idx + 1}/{len(dirs)}  {s.start:%Y-%m-%d %H:%M:%S}'
                                      + ('（監測中）' if s.running else ''))
        y = 4
        # 統計卡片
        items = list(s.summary().items())
        cols, gap, ch = 5, 8, 72
        cw = (W - (cols - 1) * gap) // cols
        for i, (k, v) in enumerate(items):
            x0, y0 = (i % cols) * (cw + gap), y + (i // cols) * (ch + gap)
            c.create_rectangle(x0, y0, x0 + cw, y0 + ch, fill=C_PANEL, outline='')
            c.create_text(x0 + 14, y0 + 18, text=k, fill=C_MUTED, font=(FONT, 12), anchor='w')
            c.create_text(x0 + 14, y0 + 48, text=v, fill=C_TEXT, font=(FONT, 18, 'bold'), anchor='w')
        y += ((len(items) + cols - 1) // cols) * (ch + gap) + 10

        # 水滴數曲線
        c.create_text(0, y + 12, text='水滴數時間曲線', fill=C_ACCENT, font=(FONT, 16, 'bold'), anchor='w')
        c.create_text(W, y + 12, text='藍線：水滴數　紅虛線：WET 門檻　紅底：WET 期間', fill=C_MUTED,
                      font=(FONT, 12), anchor='e')
        y += 30
        gh = 240
        self._draw_chart(c, 0, y, W, gh, s)
        y += gh + 18

        # 事件清單
        c.create_text(0, y + 12, text=f'WET 事件（{len(s.events)}）', fill=C_ACCENT, font=(FONT, 16, 'bold'),
                      anchor='w')
        y += 32
        if not s.events:
            c.create_text(0, y + 16, text='監測期間沒有 WET 事件', fill=C_MUTED, font=(FONT, 15), anchor='w')
            y += 44
        tw = 330
        shown = s.events[-REPORT_MAX_EVENTS:]
        if len(shown) < len(s.events):
            c.create_text(0, y + 12, text=f'只列出最近 {len(shown)} 個事件，全部事件請按「匯出並開啟 HTML」',
                          fill=C_MUTED, font=(FONT, 14), anchor='w')
            y += 36
        for ev in shown:
            img = s.event_image(ev)
            th = int(img.shape[0] * tw / img.shape[1]) if img is not None else 160
            c.create_rectangle(0, y, W, y + th + 12, fill=C_PANEL, outline='')
            if img is not None:
                photo = to_photo(img, tw, th)
                self._report_photos.append(photo)
                c.create_image(6, y + 6, image=photo, anchor='nw')
            t0 = datetime.fromtimestamp(s.start.timestamp() + ev['start'])
            c.create_text(tw + 30, y + 34, text=f"事件 {ev['id']}", fill=C_TEXT, font=(FONT, 20, 'bold'), anchor='w')
            c.create_text(tw + 30, y + 64, anchor='nw', fill=C_TEXT, font=(FONT, 15),
                          text=f"開始：{t0:%H:%M:%S}\n持續：{fmt_duration(ev['end'] - ev['start'])}\n"
                               f"最大水滴數：{ev['max_drops']}")
            y += th + 22

        # 參數
        c.create_text(0, y + 12, text='偵測參數', fill=C_ACCENT, font=(FONT, 16, 'bold'), anchor='w')
        y += 34
        lines = param_lines(s.params, s.rois)
        for i, (k, v) in enumerate(lines):
            c.create_text((i % 3) * (W // 3), y + (i // 3) * 30, text=f'{k}：{v}', fill=C_TEXT,
                          font=(FONT, 13), anchor='nw')
        y += ((len(lines) + 2) // 3) * 30 + 24
        c.configure(scrollregion=(0, 0, W, y))
        c.yview_moveto(0)

    def _draw_chart(self, c, x0, y0, w, h, s):
        c.create_rectangle(x0, y0, x0 + w, y0 + h, fill=C_PANEL, outline='')
        pl, pr, pt, pb = 56, 16, 14, 30
        pw, ph = w - pl - pr, h - pt - pb
        dur = max(s.elapsed(), 1e-6)
        thresh = s.params.get('wet_count', DETECT_DEFAULTS['wet_count'])
        recs = s.series
        ymax = max([r[1] for r in recs] + [thresh]) * 1.1 or 1
        X = lambda t: x0 + pl + t / dur * pw
        Y = lambda v: y0 + pt + ph - v / ymax * ph
        for ev in s.events:
            c.create_rectangle(X(ev['start']), y0 + pt, max(X(ev['end']), X(ev['start']) + 2), y0 + pt + ph,
                               fill='#5b2130', outline='')
        for i in range(5):
            v = ymax * i / 4
            c.create_line(x0 + pl, Y(v), x0 + pl + pw, Y(v), fill=C_CARD)
            c.create_text(x0 + pl - 8, Y(v), text=f'{v:.0f}', fill=C_MUTED, font=(FONT, 11), anchor='e')
        for i in range(6):
            t = dur * i / 5
            c.create_text(X(t), y0 + h - 13, text=fmt_duration(t), fill=C_MUTED, font=(FONT, 11))
        c.create_line(x0 + pl, Y(thresh), x0 + pl + pw, Y(thresh), fill='#f87171', dash=(6, 4))
        if len(recs) > 1:
            bucket = max(1, len(recs) // pw)
            pts = []
            for i in range(0, len(recs), bucket):
                chunk = recs[i:i + bucket]
                pts += [X(chunk[0][0]), Y(max(r[1] for r in chunk))]
            if len(pts) >= 4:
                c.create_line(*pts, fill='#38bdf8', width=2)

    def export_report(self):
        s = self._report_session(self.report_dir) if self.report_dir else None
        if s is None:
            return
        with self.lock:
            path = s.export()
        self.report_title.config(text=f'已匯出：{path.relative_to(OUTPUT_DIR.parent)}')
        try:
            subprocess.Popen(['xdg-open', str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            print(f'[警告] 無法開啟瀏覽器：{e}')

    # ── 偵測 + 顯示 ──
    def _detect(self, img, p):
        t0 = time.perf_counter()
        s = float(p['scale'])
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray_full
        if self.rois:
            rois = [tuple(int(round(v * s)) for v in r) for r in self.rois]
        else:
            rois = [auto_roi_rect(gray)]
        # 面積與核大小以原圖像素為準，縮小後換算
        min_a, max_a = max(1, p['min_area'] * s * s), p['max_area'] * s * s
        all_blobs = []
        for x1, y1, x2, y2 in rois:
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                all_blobs.append([])
                continue
            if p['method'] == 'B':
                binary = detect_method_b(roi, p['sigma'], max(3, p['kernel'] * s))
            else:
                binary = detect_method_a(roi, int(p['thresh']))
            all_blobs.append(find_blobs(binary, min_a, max_a))
        n = sum(len(b) for b in all_blobs)

        x1, y1, x2, y2 = rois[0]
        roi0 = gray[y1:y2, x1:x2]
        hist = gray_hist(roi0 if roi0.size else gray)
        d = chi2_dist(self.prev_hist, hist) if self.prev_hist is not None else 0.0
        self.prev_hist = hist
        raw_wet = d > p['hist_thresh'] or n > p['wet_count']

        # 連續 wet_frames 張與目前狀態相反才切換
        if raw_wet != self.is_wet:
            self._streak += 1
            if self._streak >= p['wet_frames']:
                self.is_wet, self._streak = raw_wet, 0
        else:
            self._streak = 0

        vis = img.copy()
        lw = max(2, img.shape[1] // 600)
        for i, ((x1, y1, x2, y2), blobs) in enumerate(zip(rois, all_blobs)):
            c = ROI_COLORS[i % len(ROI_COLORS)]
            cv2.rectangle(vis, (int(x1 / s), int(y1 / s)), (int(x2 / s), int(y2 / s)), c, lw)
            cv2.putText(vis, f'ROI {i + 1}: {len(blobs)}', (int(x1 / s) + 8, int(y1 / s) + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, c, lw + 1, cv2.LINE_AA)
            for cx, cy, area in blobs:
                rad = max(6, int(np.sqrt(area / np.pi) / s) + 4)
                cv2.circle(vis, (int((cx + x1) / s), int((cy + y1) / s)), rad, (0, 200, 255), lw)
        text = f'{datetime.now():%Y-%m-%d %H:%M:%S}  {"WET" if self.is_wet else "DRY"}  drops={n}'
        org = (20, img.shape[0] - 24)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), lw + 4, cv2.LINE_AA)
        cv2.putText(vis, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (0, 60, 255) if self.is_wet else (60, 220, 60), lw, cv2.LINE_AA)
        self.proc_ms = (time.perf_counter() - t0) * 1000
        return vis, n, d

    def _process_loop(self):
        """背景執行緒：解拜耳 → 偵測 → 寫紀錄 → 縮成畫布大小。OpenCV 運算時會釋放 GIL。"""
        last = 0
        while self.running:
            if not self.grabber.new_frame.wait(0.5):
                continue
            self.grabber.new_frame.clear()
            with self.grabber.lock:
                latest = self.grabber.latest
            if not latest or latest[3] == last or self.paused:
                continue
            data, w, h, last = latest
            try:
                frame = to_bgr(data, w, h, self.grabber.pixel_format)
                vis, n, chi2 = self._detect(frame, dict(self.params))
                with self.lock:
                    if self.session and self.session.running:
                        self.session.add(n, self.is_wet, chi2, vis)
                view = self._fit_view(vis) if self.page == 'live' else None
                self.result = (frame, vis, n, chi2, view, last)
            except Exception as e:     # 單張失敗不要讓執行緒整個停掉
                print(f'[警告] 處理影像失敗：{e!r}')

    def _tick(self):
        res = self.result
        if res is not None and res[-1] != self.last_seq:
            self.last_seq = res[-1]
            self.frame_bgr, self.vis_bgr, self.n_drops, self.chi2, view = res[:5]
            if self.page == 'live':
                self._put_view(view or self._fit_view(self.vis_bgr))

        color = C_RED if self.is_wet else C_GREEN
        self.status_lbl.config(text='WET' if self.is_wet else 'DRY', bg=color)
        self.count_lbl.config(text=str(self.n_drops))
        mon = ''
        if self.session and self.session.running:
            mon = f'\n● 監測中 {fmt_duration(self.session.elapsed())}　事件 {len(self.session.events)}'
        self.info_lbl.config(
            text=f'相機 {self.grabber.fps():.1f} fps　偵測 {self.proc_ms:.0f} ms　壞影格 {self.grabber.n_bad}'
                 + ('　【暫停】' if self.paused else '') + mon)
        self.root.after(30, self._tick)

    def _on_canvas_resize(self, e):
        if e.width >= 10 and e.height >= 10:
            self.view_size = (e.width, e.height)
            if self.vis_bgr is not None:
                self._put_view(self._fit_view(self.vis_bgr))

    def _fit_view(self, img):
        """依畫布大小等比例縮放置中，切成橫帶。回傳 (橫帶清單, 寬, 高, 縮放比例, 左上角位移)。不碰 Tk。"""
        h, w = img.shape[:2]
        cw, ch = self.view_size
        s = min(cw / w, ch / h)
        vw, vh = int(w * s), int(h * s)
        if ImageTk is None:
            return [to_ppm(img, vw, vh)], vw, vh, s, ((cw - vw) // 2, (ch - vh) // 2)
        rgb = cv2.cvtColor(cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        bounds = [vh * i // VIEW_STRIPS for i in range(VIEW_STRIPS + 1)]
        strips = [Image.fromarray(rgb[y0:y1]) for y0, y1 in zip(bounds, bounds[1:])]
        return strips, vw, vh, s, ((cw - vw) // 2, (ch - vh) // 2)

    def _put_view(self, view):
        strips, vw, vh, self.view_scale, self.view_offset = view
        if ImageTk is None:
            self._photos = [tk.PhotoImage(width=vw, height=vh, data=strips[0], format='PPM')]
            self.canvas.delete('img')
            self.canvas.create_image(*self.view_offset, anchor='nw', image=self._photos[0], tags='img')
            self.canvas.tag_lower('img')
            return
        if self._view_key != (vw, vh, self.view_offset):
            # 尺寸改變：重建每條橫帶的 PhotoImage 與畫布物件
            self._view_key = (vw, vh, self.view_offset)
            if self._strip_job is not None:
                self.root.after_cancel(self._strip_job)
                self._strip_job = None
            self._photos = [ImageTk.PhotoImage(im) for im in strips]
            self.canvas.delete('img')
            y = self.view_offset[1]
            for ph in self._photos:
                self.canvas.create_image(self.view_offset[0], y, anchor='nw', image=ph, tags='img')
                y += ph.height()
            self.canvas.tag_lower('img')
            return
        self._pending_strips = strips   # 之後由 _paste_strip 每次貼一條
        if self._strip_job is None:
            self._paste_strip(0)

    def _paste_strip(self, i):
        strips = self._pending_strips
        self._photos[i].paste(strips[i])
        if i + 1 < len(strips):
            self._strip_job = self.root.after(1, self._paste_strip, i + 1)
        else:
            self._strip_job = None

    def save(self):
        if self.frame_bgr is None:
            return
        OUTPUT_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(str(OUTPUT_DIR / f'live_{stamp}_raw.jpg'), self.frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(OUTPUT_DIR / f'live_{stamp}_det.jpg'), self.vis_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.info_lbl.config(text=f'已存檔：output/live_{stamp}_raw.jpg、_det.jpg')

    def close(self):
        try:
            self.running = False
            self.worker.join(timeout=3)
            if self.session and self.session.running:
                self.session.stop()
                self.session.export()
            self._save_settings()
        finally:
            self.grabber.stop()
            self.root.destroy()


def main():
    ap = argparse.ArgumentParser(description='即時水滴偵測（觸控版）')
    ap.add_argument('--fullscreen', action='store_true', help='全螢幕開啟')
    args = ap.parse_args()

    try:
        camera = open_camera(None)
    except SystemExit:
        # 桌面捷徑啟動時看不到終端機，改用視窗提示，避免點了像沒反應
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            '水滴偵測',
            '找不到相機，無法開啟。\n\n'
            '請確認：\n'
            '1. 相機網路線插在 Pi 的網口，且相機有電\n'
            '2. 沒有其他電腦（廠商軟體）正在使用這台相機\n\n'
            '詳細訊息：output/live_app.log')
        root.destroy()
        return 2
    try_call(camera.clear_triggers)
    model = try_call(camera.get_model_name, default='')
    pixel_format = pixel_format_for(model)
    try:
        camera.set_pixel_format_from_string(pixel_format)
    except GLib.Error as e:
        print(f'[警告] 設定像素格式 {pixel_format} 失敗：{e.message}')
    if camera.is_gv_device():
        try:
            camera.gv_auto_packet_size()
        except GLib.Error as e:
            print(f'[警告] 自動協商封包大小失敗：{e.message}')
    root = tk.Tk()
    LiveDropApp(root, camera, fullscreen=args.fullscreen)
    root.mainloop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
