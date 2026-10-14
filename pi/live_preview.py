#!/usr/bin/env python3
"""
GigE 相機即時預覽（Raspberry Pi 5 + Aravis）
==========================================
用途：調整鏡頭對焦、曝光與白平衡；之後再接水滴偵測。

畫面資訊：fps、曝光、增益、對焦分數（中央框 Laplacian 變異數，越大越清楚）、
中央框 B/G/R 平均值（放白紙時三者應接近）。

按鍵：
  q / Esc   離開
  z         切換 中央 1:1 放大（對焦用）/ 全畫面
  + / -     曝光 ×1.25 / ÷1.25
  ] / [     增益 +1 / -1 dB
  w         以中央框（白紙或灰卡）校正白平衡，可多按幾次逼近
  s         存一張全解析度影像到 output/preview_<時間>.jpg
  h         顯示 / 隱藏說明

用法：
  python3 live_preview.py                  # 第一台相機，10 fps
  python3 live_preview.py --fps 5 --exposure 30000
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from aravis_capture_test import Aravis, GLib, open_camera, to_bgr, try_call
import cv2
import numpy as np

WINDOW = 'GigE Live Preview'
BOX_FRAC = 0.2          # 中央框邊長佔畫面比例
VIEW_MAX_W = 1280       # 全畫面顯示寬度


def parse_args():
    ap = argparse.ArgumentParser(description='GigE 相機即時預覽')
    ap.add_argument('--camera', default=None, help='相機名稱（預設第一台）')
    ap.add_argument('--fps', type=float, default=10.0, help='幀率（預設 10）')
    ap.add_argument('--exposure', type=float, default=None, help='曝光時間（µs）')
    ap.add_argument('--gain', type=float, default=None, help='增益（dB）')
    ap.add_argument('--cv-pattern', choices=['BG', 'RG', 'GB', 'GR'], default=None,
                    help='強制指定 OpenCV Bayer 轉換碼')
    return ap.parse_args()


def setup(camera, args):
    try_call(camera.clear_triggers)
    if args.exposure is not None:
        try_call(camera.set_exposure_time_auto, Aravis.Auto.OFF)
        try_call(camera.set_exposure_time, args.exposure)
    if args.gain is not None:
        try_call(camera.set_gain_auto, Aravis.Auto.OFF)
        try_call(camera.set_gain, args.gain)
    try_call(camera.set_frame_rate, args.fps)
    if camera.is_gv_device():
        try:
            print(f'自動協商封包大小：{camera.gv_auto_packet_size()} bytes')
        except GLib.Error as e:
            print(f'[警告] 自動協商封包大小失敗：{e.message}')


def clamp_set(setter, bounds_fn, value):
    lo, hi = bounds_fn()
    value = min(max(value, lo), hi)
    try:
        setter(value)
    except GLib.Error as e:
        print(f'[警告] {e.message}')
    return value


def white_balance(camera, box_bgr):
    """以中央框平均值調整 BalanceRatio：R、B 往 G 靠攏。"""
    b, g, r = (float(box_bgr[..., i].mean()) for i in range(3))
    if min(b, g, r) < 5 or max(b, g, r) > 250:
        print(f'[白平衡] 中央框太暗或過曝（B={b:.0f} G={g:.0f} R={r:.0f}），請調整曝光後再按 w')
        return
    for channel, mean in (('Red', r), ('Blue', b)):
        try:
            camera.set_string('BalanceRatioSelector', channel)
            old = camera.get_integer('BalanceRatio')
            lo, hi = camera.get_integer_bounds('BalanceRatio')
            new = int(min(max(round(old * g / mean), lo), hi))
            camera.set_integer('BalanceRatio', new)
            print(f'[白平衡] {channel}: {old} → {new}')
        except GLib.Error as e:
            print(f'[白平衡] {channel} 失敗：{e.message}')


def draw_lines(img, lines, origin=(10, 30), scale=0.8):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2, cv2.LINE_AA)
        y += int(32 * scale / 0.8)


HELP = ['q:quit  z:zoom 1:1  +/-:exposure  ]/[:gain',
        'w:white balance (white paper in box)  s:save  h:help']


def main():
    args = parse_args()
    camera = open_camera(args.camera)
    setup(camera, args)
    pixel_format = camera.get_pixel_format_as_string()

    stream = camera.create_stream(None, None)
    if stream is None:
        print('[錯誤] 無法建立串流')
        return 2
    if camera.is_gv_device():
        stream.set_property('socket-buffer', Aravis.GvStreamSocketBuffer.AUTO)
        stream.set_property('packet-resend', Aravis.GvStreamPacketResend.ALWAYS)
    payload = camera.get_payload()
    for _ in range(6):
        stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    sized = False
    camera.start_acquisition()
    print(f'開始預覽：{camera.get_model_name()} {pixel_format} @ {args.fps} fps，按 q 離開')

    zoom, show_help = False, True
    frame_times = []
    n_bad = 0
    img = None
    try:
        while True:
            buf = stream.timeout_pop_buffer(1_000_000)
            # 只顯示最新一張，舊的直接還回去
            while buf is not None:
                newer = stream.try_pop_buffer()
                if newer is None:
                    break
                stream.push_buffer(buf)
                buf = newer

            if buf is not None:
                ok = buf.get_status() == Aravis.BufferStatus.SUCCESS
                if ok:
                    data = buf.get_data()
                    w, h = buf.get_image_width(), buf.get_image_height()
                stream.push_buffer(buf)
                if ok and len(data) >= w * h:
                    img = to_bgr(data, w, h, pixel_format, args.cv_pattern)
                    frame_times.append(time.perf_counter())
                    frame_times = frame_times[-20:]
                else:
                    n_bad += 1

            if img is not None:
                h, w = img.shape[:2]
                bw, bh = int(w * BOX_FRAC), int(h * BOX_FRAC)
                x0, y0 = (w - bw) // 2, (h - bh) // 2
                box = img[y0:y0 + bh, x0:x0 + bw]
                focus = cv2.Laplacian(cv2.cvtColor(box, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                b, g, r = (box[..., i].mean() for i in range(3))
                fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                       if len(frame_times) > 1 else 0.0)

                if zoom:
                    vw = min(VIEW_MAX_W, w)
                    vh = int(vw * h / w)
                    cx, cy = w // 2, h // 2
                    view = img[cy - vh // 2:cy + vh // 2, cx - vw // 2:cx + vw // 2].copy()
                else:
                    view = cv2.resize(img, (VIEW_MAX_W, int(VIEW_MAX_W * h / w)),
                                      interpolation=cv2.INTER_AREA)
                    s = VIEW_MAX_W / w
                    cv2.rectangle(view, (int(x0 * s), int(y0 * s)),
                                  (int((x0 + bw) * s), int((y0 + bh) * s)), (0, 255, 0), 2)

                exp = try_call(camera.get_exposure_time, default=0) or 0
                gain = try_call(camera.get_gain, default=0) or 0
                lines = [f'{fps:4.1f} fps  exp {exp / 1000:.1f} ms  gain {gain:.1f} dB  bad {n_bad}'
                         f'{"  [ZOOM 1:1]" if zoom else ""}',
                         f'focus {focus:8.1f}   box B/G/R {b:5.1f} {g:5.1f} {r:5.1f}']
                draw_lines(view, lines + (HELP if show_help else []))
                cv2.imshow(WINDOW, view)
                if not sized:
                    cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                    sized = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('z'):
                zoom = not zoom
            elif key == ord('h'):
                show_help = not show_help
            elif key in (ord('+'), ord('=')):
                try_call(camera.set_exposure_time_auto, Aravis.Auto.OFF)
                clamp_set(camera.set_exposure_time, camera.get_exposure_time_bounds,
                          camera.get_exposure_time() * 1.25)
            elif key in (ord('-'), ord('_')):
                try_call(camera.set_exposure_time_auto, Aravis.Auto.OFF)
                clamp_set(camera.set_exposure_time, camera.get_exposure_time_bounds,
                          camera.get_exposure_time() / 1.25)
            elif key in (ord(']'), ord('[')):
                try_call(camera.set_gain_auto, Aravis.Auto.OFF)
                clamp_set(camera.set_gain, camera.get_gain_bounds,
                          camera.get_gain() + (1 if key == ord(']') else -1))
            elif key == ord('w') and img is not None:
                white_balance(camera, box)
            elif key == ord('s') and img is not None:
                out = Path(__file__).resolve().parent.parent / 'output'
                out.mkdir(exist_ok=True)
                path = out / f'preview_{datetime.now():%Y%m%d_%H%M%S}.jpg'
                cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f'已存檔：{path}')
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop_acquisition()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
