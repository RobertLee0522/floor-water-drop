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
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

from aravis_capture_test import Aravis, GLib, open_camera, to_bgr, try_call
import cv2
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'output'
SETTINGS_FILE = OUTPUT_DIR / 'live_settings.json'
SESSIONS_DIR = OUTPUT_DIR / 'sessions'

WIN_W, WIN_H = 1280, 800
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
# Jinghang JHEM600GC 感測器實際排列是 G B / R G；設成其他 Bayer 格式時相機只改標籤、
# 資料排列不變，解拜耳會整張偏綠，所以固定用 BayerGB8
PIXEL_FORMAT = 'BayerGB8'

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
ROW_THRESH_RATIO = 1.3
ROI_PADDING = 30
ROI_COLORS = [(0, 220, 255), (255, 120, 0), (180, 0, 255), (0, 255, 120), (255, 0, 160)]
SNAPSHOT_W = 1280           # 事件截圖寬度

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


def to_photo(img_bgr, w, h):
    """BGR 影像縮放後轉成 Tk PhotoImage（用 PPM，不需要 PIL）。"""
    small = cv2.cvtColor(cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return tk.PhotoImage(width=w, height=h, data=f'P6 {w} {h} 255 '.encode() + small.tobytes(),
                         format='PPM')


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
    """一次監測：每張影像的紀錄 + WET 事件 + 截圖。"""

    def __init__(self, params, rois):
        self.start = datetime.now()
        self.end = None
        self.dir = SESSIONS_DIR / self.start.strftime('%Y%m%d_%H%M%S')
        (self.dir / 'events').mkdir(parents=True, exist_ok=True)
        self.params = dict(params)
        self.rois = list(rois)
        self.records = []           # (經過秒數, 水滴數, wet, chi2)
        self.events = []            # dict: id, start, end, max_drops, image(記憶體), file
        self._open = None

    @property
    def running(self):
        return self.end is None

    def elapsed(self):
        return ((self.end or datetime.now()) - self.start).total_seconds()

    def add(self, n, wet, chi2, vis_bgr):
        t = self.elapsed()
        self.records.append((t, n, wet, chi2))
        if wet and self._open is None:
            self._open = {'id': len(self.events) + 1, 'start': t, 'end': t, 'max_drops': -1,
                          'image': None, 'file': None}
            self.events.append(self._open)
        if self._open is not None:
            ev = self._open
            ev['end'] = t
            if n > ev['max_drops']:
                ev['max_drops'] = n
                h, w = vis_bgr.shape[:2]
                ev['image'] = cv2.resize(vis_bgr, (SNAPSHOT_W, int(h * SNAPSHOT_W / w)),
                                         interpolation=cv2.INTER_AREA)
            if not wet:
                self._write_snapshot(ev)
                self._open = None

    def _write_snapshot(self, ev):
        if ev['image'] is not None:
            ev['file'] = f"events/event_{ev['id']:03d}.jpg"
            cv2.imwrite(str(self.dir / ev['file']), ev['image'], [cv2.IMWRITE_JPEG_QUALITY, 88])

    def stop(self):
        if self._open is not None:
            self._write_snapshot(self._open)
            self._open = None
        self.end = datetime.now()

    def summary(self):
        recs = self.records
        dur = self.elapsed()
        drops = [r[1] for r in recs]
        n_wet = sum(1 for r in recs if r[2])
        wet_time = sum(e['end'] - e['start'] for e in self.events)
        return {
            '開始時間': self.start.strftime('%Y-%m-%d %H:%M:%S'),
            '結束時間': self.end.strftime('%Y-%m-%d %H:%M:%S') if self.end else '監測中',
            '監測時間': fmt_duration(dur),
            '處理影像數': f'{len(recs)}',
            '平均處理速度': f'{len(recs) / dur:.1f} fps' if dur > 0 else '-',
            'WET 事件數': f'{len(self.events)}',
            'WET 影像比例': f'{n_wet / len(recs):.1%}' if recs else '-',
            'WET 累計時間': fmt_duration(wet_time),
            '最大水滴數': f'{max(drops)}' if drops else '-',
            '平均水滴數': f'{np.mean(drops):.1f}' if drops else '-',
        }

    def export(self):
        """寫出 frames.csv 與 report.html，回傳 report.html 路徑。"""
        if self._open is not None:
            self._write_snapshot(self._open)     # 進行中的事件先存一張目前的截圖
        with open(self.dir / 'frames.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['elapsed_s', 'time', 'drops', 'wet', 'hist_chi2'])
            for t, n, wet, d in self.records:
                ts = datetime.fromtimestamp(self.start.timestamp() + t).strftime('%H:%M:%S.%f')[:-3]
                w.writerow([f'{t:.3f}', ts, n, int(wet), f'{d:.4f}'])
        path = self.dir / 'report.html'
        path.write_text(render_html(self), encoding='utf-8')
        return path


# ─────────── HTML 報表 ───────────

def chart_svg(session, width=1000, height=280):
    recs = session.records
    pad_l, pad_r, pad_t, pad_b = 48, 16, 14, 34
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    dur = max(session.elapsed(), 1e-6)
    thresh = session.params.get('wet_count', 30)
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
        lines += [(label, params.get(key)) for key, label, *_ in group]
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
        self.session = None
        self.page = 'live'
        self.group = '相機'

        self._build_ui()
        self._load_settings()
        if fullscreen:
            self.toggle_fullscreen()
        self.grabber.start()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
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
        self.canvas = tk.Canvas(page, width=VIEW_W, height=VIEW_H, bg='black', highlightthickness=0)
        self.canvas.pack(side='left', padx=6, pady=6)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        panel = tk.Frame(page, bg=C_PANEL, width=PANEL_W)
        panel.pack(side='left', fill='y', pady=6)
        panel.pack_propagate(False)

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
                b = self._btn(row, text, lambda v=val: self._set_param('method', v), size=14)
                b.pack(side='left', expand=True, fill='x', padx=2, ipady=10)
                self.method_btns[val] = b
            self._refresh_method()
        for key, label, lo, hi, step, digits in PARAM_GROUPS[name]:
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
        if name == '相機':
            self._btn(f, '自動白平衡（畫面中央放白紙）', self.auto_wb, size=14).pack(fill='x', pady=6, ipady=10)
        self._refresh_values()

    def _refresh_values(self):
        for key, lbl in self.value_lbls.items():
            lbl.config(text=f'{self.params[key]:.{self.specs[key][5]}f}')

    def _refresh_method(self):
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
        except (OSError, ValueError):
            pass
        self._apply_all(params, force=True)
        self._update_roi_label()

    def _save_settings(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps({'params': self.params, 'rois': self.rois},
                                            ensure_ascii=False, indent=2))

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
        if self.session and self.session.running:
            self.session.stop()
            self.session.export()
            self.mon_btn.config(text='開始監測', bg=C_GREEN, activebackground=C_GREEN)
            if self.page == 'report':
                self._render_report()
        else:
            self.session = Session(self.params, self.rois)
            self.mon_btn.config(text='停止監測', bg=C_RED, activebackground=C_RED)

    def toggle_page(self):
        if self.page == 'live':
            self.page = 'report'
            self.live_page.pack_forget()
            self.report_page.pack(fill='both', expand=True)
            self.page_btn.config(text='即時')
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

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text='繼續' if self.paused else '暫停')

    # ── 報表頁 ──
    def _build_report_page(self):
        page = self.report_page = tk.Frame(self.body, bg=C_BG)
        bar = tk.Frame(page, bg=C_BG)
        bar.pack(fill='x', padx=8, pady=(8, 6))
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

    def _render_report(self):
        c = self.rcanvas
        c.delete('all')
        self._report_photos = []
        s = self.session
        W = WIN_W - 16
        if s is None:
            self.report_title.config(text='尚無監測紀錄')
            c.create_text(W // 2, 180, text='按上方「開始監測」開始記錄，\n之後回到這頁就能預覽報表。',
                          fill=C_MUTED, font=(FONT, 20), justify='center')
            c.configure(scrollregion=(0, 0, W, 400))
            return
        self.report_title.config(text=f'監測紀錄  {s.start:%Y-%m-%d %H:%M:%S}' + ('（監測中）' if s.running else ''))
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
        for ev in s.events:
            th = int(ev['image'].shape[0] * tw / ev['image'].shape[1]) if ev['image'] is not None else 160
            c.create_rectangle(0, y, W, y + th + 12, fill=C_PANEL, outline='')
            if ev['image'] is not None:
                photo = to_photo(ev['image'], tw, th)
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
        thresh = s.params['wet_count']
        recs = s.records
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
        if self.session is None:
            return
        path = self.session.export()
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

    def _tick(self):
        with self.grabber.lock:
            latest = self.grabber.latest
        if latest and latest[3] != self.last_seq and not self.paused:
            data, w, h, seq = latest
            self.last_seq = seq
            self.frame_bgr = to_bgr(data, w, h, self.grabber.pixel_format)
            self.vis_bgr, self.n_drops, self.chi2 = self._detect(self.frame_bgr, self.params)
            if self.session and self.session.running:
                self.session.add(self.n_drops, self.is_wet, self.chi2, self.vis_bgr)
            if self.page == 'live':
                self._show(self.vis_bgr)

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

    def _show(self, img):
        h, w = img.shape[:2]
        s = min(VIEW_W / w, VIEW_H / h)
        vw, vh = int(w * s), int(h * s)
        self.view_scale = s
        self.view_offset = ((VIEW_W - vw) // 2, (VIEW_H - vh) // 2)
        self._photo = to_photo(img, vw, vh)
        self.canvas.delete('img')
        self.canvas.create_image(*self.view_offset, anchor='nw', image=self._photo, tags='img')
        self.canvas.tag_lower('img')

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

    camera = open_camera(None)
    try_call(camera.clear_triggers)
    try:
        camera.set_pixel_format_from_string(PIXEL_FORMAT)
    except GLib.Error as e:
        print(f'[警告] 設定像素格式 {PIXEL_FORMAT} 失敗：{e.message}')
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
