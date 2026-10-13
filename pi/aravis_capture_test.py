#!/usr/bin/env python3
"""
Aravis GigE 擷取測試（Raspberry Pi 5）
======================================
用 Aravis（PyGObject）連續擷取影像，量測：
  - 實際擷取 fps、完成 / 失敗影格、遺失與重送封包（Aravis stream 統計）
  - 影像取出（buffer 複製）與解拜耳的耗時，評估 Pi 上的處理餘裕
  - 程序 CPU 使用率（含 Aravis 接收執行緒）
並存下前幾張解拜耳後的影像，用來確認顏色 / Bayer 排列是否正確。

用法：
  python3 aravis_capture_test.py --fake --duration 5        # 不接相機，驗證安裝
  python3 aravis_capture_test.py --probe                    # 只列出相機資訊與可用像素格式
  python3 aravis_capture_test.py --fps 10 --duration 20     # 擷取測試
  python3 aravis_capture_test.py --fps 15 --roi 0,512,3072,1024 --save 5

結束代碼：0 = 達標（成功率 >= 99% 且 fps >= 目標 90%），1 = 未達標，2 = 無法開啟相機
"""

import argparse
import csv
import glob
import json
import os
import socket
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


def _ensure_typelib_path():
    """從原始碼安裝到 /usr/local 時，PyGObject 預設找不到 typelib；在 import gi 前補上路徑。"""
    found = glob.glob('/usr/local/lib/*/girepository-1.0') + glob.glob('/usr/local/lib/girepository-1.0')
    current = [p for p in os.environ.get('GI_TYPELIB_PATH', '').split(':') if p]
    extra = [p for p in found if p not in current]
    if extra:
        os.environ['GI_TYPELIB_PATH'] = ':'.join(extra + current)


_ensure_typelib_path()

import gi  # noqa: E402
gi.require_version('Aravis', '0.8')
from gi.repository import Aravis, GLib, GObject  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ─────────── 像素格式 ───────────
# GenICam 以左上角 2x2 命名（BayerRG = R G / G B）；OpenCV 的舊命名以第二列的
# 第二、三個像素為準，因此兩者是「錯開」的：GenICam BayerRG8 → COLOR_BayerBG2BGR。
BAYER_TO_CV = {
    'BayerRG8': cv2.COLOR_BayerBG2BGR,
    'BayerBG8': cv2.COLOR_BayerRG2BGR,
    'BayerGR8': cv2.COLOR_BayerGB2BGR,
    'BayerGB8': cv2.COLOR_BayerGR2BGR,
}
# 自動選擇時的優先順序：Bayer8（頻寬最小的彩色）→ Mono8
FORMAT_PRIORITY = ['BayerRG8', 'BayerBG8', 'BayerGR8', 'BayerGB8', 'Mono8']

STAT_KEYS = [
    'n_completed_buffers', 'n_failures', 'n_underruns', 'n_timeouts', 'n_aborted',
    'n_missing_frames', 'n_size_mismatch_errors', 'n_received_packets',
    'n_missing_packets', 'n_error_packets', 'n_ignored_packets', 'n_resend_requests',
    'n_resent_packets', 'n_resend_ratio_reached', 'n_resend_disabled',
    'n_duplicated_packets', 'n_transferred_bytes', 'n_ignored_bytes',
]


def to_bgr(data, width, height, pixel_format, cv_pattern=None):
    """把 buffer 原始資料轉成 OpenCV BGR 影像。cv_pattern 可強制指定 OpenCV 的 Bayer 轉換碼。"""
    raw = np.frombuffer(data, dtype=np.uint8)
    if pixel_format in BAYER_TO_CV:
        code = getattr(cv2, f'COLOR_Bayer{cv_pattern}2BGR') if cv_pattern else BAYER_TO_CV[pixel_format]
        return cv2.cvtColor(raw[:width * height].reshape(height, width), code)
    if pixel_format == 'Mono8':
        return cv2.cvtColor(raw[:width * height].reshape(height, width), cv2.COLOR_GRAY2BGR)
    if pixel_format == 'RGB8':
        return cv2.cvtColor(raw[:width * height * 3].reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if pixel_format == 'BGR8':
        return raw[:width * height * 3].reshape(height, width, 3).copy()
    raise ValueError(f'不支援的像素格式：{pixel_format}（請改用 Bayer*8 或 Mono8）')


def packet_socket_allowed():
    """與 Aravis 相同的判斷方式：能開 AF_PACKET raw socket 才會走 packet socket 模式。"""
    try:
        s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
        s.close()
        return True
    except (PermissionError, OSError, AttributeError):
        return False


def try_call(fn, *a, default=None):
    try:
        return fn(*a)
    except (GLib.Error, TypeError, AttributeError):
        return default


def read_stream_stats(stream):
    stats = {}
    try:
        for i in range(stream.get_n_infos()):
            if stream.get_info_type(i) == GObject.TYPE_UINT64:
                stats[stream.get_info_name(i)] = stream.get_info_uint64(i)
    except (AttributeError, GLib.Error):
        pass
    if not stats:
        for key in STAT_KEYS:
            v = try_call(stream.get_info_uint64_by_name, key)
            if v is not None:
                stats[key] = v
    return stats


def pct(values, q):
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100 * (len(s) - 1))))]


# ─────────── 相機資訊 ───────────
def camera_info(camera):
    info = {
        'vendor': try_call(camera.get_vendor_name),
        'model': try_call(camera.get_model_name),
        'serial': try_call(camera.get_device_serial_number),
        'is_gige': camera.is_gv_device(),
        'sensor_size': try_call(camera.get_sensor_size),
        'region': try_call(camera.get_region),
        'pixel_format': try_call(camera.get_pixel_format_as_string),
        'available_formats': try_call(camera.dup_available_pixel_formats_as_strings, default=[]),
        'frame_rate_available': try_call(camera.is_frame_rate_available, default=False),
        'frame_rate_bounds': try_call(camera.get_frame_rate_bounds),
        'frame_rate': try_call(camera.get_frame_rate),
        'exposure_us': try_call(camera.get_exposure_time),
        'gain_db': try_call(camera.get_gain),
        'payload': try_call(camera.get_payload),
    }
    if info['is_gige']:
        info['packet_size'] = try_call(camera.gv_get_packet_size)
        info['packet_delay_ns'] = try_call(camera.gv_get_packet_delay)
    return info


def pick_format(available):
    for fmt in FORMAT_PRIORITY:
        if fmt in available:
            return fmt
    return None


# ─────────── 主程式 ───────────
def parse_args():
    ap = argparse.ArgumentParser(description='Aravis GigE 擷取測試')
    ap.add_argument('--camera', default=None, help='相機名稱（arv-tool-0.8 列出的 ID，預設第一台）')
    ap.add_argument('--probe', action='store_true', help='只輸出相機資訊與建議像素格式')
    ap.add_argument('--fake', action='store_true', help='使用 Aravis 內建模擬相機（不需實體相機，用來驗證安裝）')
    ap.add_argument('--fps', type=float, default=10.0, help='目標幀率（預設 10）')
    ap.add_argument('--duration', type=float, default=20.0, help='擷取秒數（預設 20）')
    ap.add_argument('--pixel-format', default='auto', help='像素格式，例如 BayerRG8；auto = 自動選 Bayer8')
    ap.add_argument('--roi', default=None, help='x,y,w,h（在相機端裁切，減少頻寬）；full = 全幅')
    ap.add_argument('--exposure', type=float, default=None, help='曝光時間（µs），未指定則不變更')
    ap.add_argument('--gain', type=float, default=None, help='增益（dB），未指定則不變更')
    ap.add_argument('--packet-size', default='auto', help='GigE 封包大小；auto = 自動協商')
    ap.add_argument('--packet-delay', type=int, default=None, help='GigE 封包間隔（ns），用於平滑頻寬')
    ap.add_argument('--no-packet-socket', action='store_true', help='強制使用一般 UDP socket 接收')
    ap.add_argument('--buffers', type=int, default=10, help='串流緩衝數（預設 10）')
    ap.add_argument('--no-debayer', action='store_true', help='不做解拜耳（只量測擷取）')
    ap.add_argument('--cv-pattern', choices=['BG', 'RG', 'GB', 'GR'], default=None,
                    help='強制指定 OpenCV Bayer 轉換碼（顏色不對時使用，不影響相機設定）')
    ap.add_argument('--save', type=int, default=3, help='存下前 N 張影像（預設 3）')
    ap.add_argument('--outdir', default=None, help='輸出資料夾（預設 output/capture_<時間>）')
    return ap.parse_args()


def open_camera(name):
    try:
        return Aravis.Camera.new(name)
    except (GLib.Error, TypeError) as e:
        print(f'[錯誤] 無法開啟相機 {name or "(第一台)"}：{e}')
        print('        請先執行 arv-tool-0.8 確認相機有被找到。')
        sys.exit(2)


def configure(camera, args):
    """套用設定，回傳實際使用的像素格式。設定失敗的項目只警告不中止。"""
    notes = []

    def attempt(label, fn, *a):
        try:
            fn(*a)
        except GLib.Error as e:
            notes.append(f'{label} 失敗：{e.message}')
            print(f'[警告] {label} 失敗：{e.message}')

    # 自由執行（關閉觸發）
    attempt('關閉觸發', camera.clear_triggers)

    available = try_call(camera.dup_available_pixel_formats_as_strings, default=[]) or []
    fmt = pick_format(available) if args.pixel_format == 'auto' else args.pixel_format
    if fmt is None:
        print(f'[警告] 找不到 Bayer8 / Mono8，可用格式：{available}；沿用目前格式')
    else:
        attempt(f'設定像素格式 {fmt}', camera.set_pixel_format_from_string, fmt)

    if args.roi == 'full':
        w, h = camera.get_sensor_size()
        attempt(f'設定全幅 ROI {w}x{h}', camera.set_region, 0, 0, w, h)
    elif args.roi:
        x, y, w, h = (int(v) for v in args.roi.split(','))
        attempt(f'設定 ROI {x},{y},{w},{h}', camera.set_region, x, y, w, h)

    if args.exposure is not None:
        attempt('關閉自動曝光', camera.set_exposure_time_auto, Aravis.Auto.OFF)
        attempt(f'設定曝光 {args.exposure} µs', camera.set_exposure_time, args.exposure)

    if args.gain is not None:
        attempt('關閉自動增益', camera.set_gain_auto, Aravis.Auto.OFF)
        attempt(f'設定增益 {args.gain} dB', camera.set_gain, args.gain)

    if try_call(camera.is_frame_rate_available, default=False):
        attempt(f'設定幀率 {args.fps}', camera.set_frame_rate, args.fps)
    else:
        print('[警告] 相機不支援設定幀率，將以相機目前速率擷取')

    if camera.is_gv_device():
        if args.packet_size == 'auto':
            try:
                size = camera.gv_auto_packet_size()
                print(f'自動協商封包大小：{size} bytes')
            except GLib.Error as e:
                print(f'[警告] 自動協商封包大小失敗：{e.message}')
        else:
            attempt(f'設定封包大小 {args.packet_size}', camera.gv_set_packet_size, int(args.packet_size))
        if args.packet_delay is not None:
            attempt(f'設定封包間隔 {args.packet_delay} ns', camera.gv_set_packet_delay, args.packet_delay)
        if args.no_packet_socket:
            camera.gv_set_stream_options(Aravis.GvStreamOption.PACKET_SOCKET_DISABLED)

    return try_call(camera.get_pixel_format_as_string) or fmt, notes


def main():
    args = parse_args()
    if args.fake:
        Aravis.enable_interface('Fake')
        args.camera = args.camera or 'Fake_1'
    camera = open_camera(args.camera)

    if args.probe:
        info = camera_info(camera)
        print(json.dumps(info, ensure_ascii=False, indent=2, default=str))
        best = pick_format(info['available_formats'] or [])
        print(f'BEST_FORMAT={best or ""}')
        size = info['sensor_size']
        print(f'SENSOR_SIZE={size[0]}x{size[1]}' if size else 'SENSOR_SIZE=')
        return 0

    outdir = Path(args.outdir or f'output/capture_{datetime.now():%Y%m%d_%H%M%S}')
    outdir.mkdir(parents=True, exist_ok=True)

    pixel_format, notes = configure(camera, args)
    info = camera_info(camera)
    is_gige = info['is_gige']
    use_packet_socket = is_gige and not args.no_packet_socket and packet_socket_allowed()

    print('─' * 60)
    print(f"相機      : {info['vendor']} {info['model']} (SN {info['serial']})")
    print(f"ROI       : {info['region']}  像素格式 {pixel_format}")
    print(f"幀率      : 目標 {args.fps}，相機回報 {info['frame_rate']}，範圍 {info['frame_rate_bounds']}")
    payload = camera.get_payload()
    print(f"Payload   : {payload} bytes ≈ {payload * args.fps * 8 / 1e6:.0f} Mbps @ {args.fps} fps")
    if is_gige:
        print(f"封包大小  : {info.get('packet_size')} bytes")
        print(f"接收模式  : {'packet socket（AF_PACKET）' if use_packet_socket else 'UDP socket'}")
    print('─' * 60)

    stream = camera.create_stream(None, None)
    if stream is None:
        print('[錯誤] 無法建立串流')
        return 2
    if is_gige:
        stream.set_property('socket-buffer', Aravis.GvStreamSocketBuffer.AUTO)
        stream.set_property('socket-buffer-size', 0)
        stream.set_property('packet-resend', Aravis.GvStreamPacketResend.ALWAYS)

    for _ in range(args.buffers):
        stream.push_buffer(Aravis.Buffer.new_allocate(payload))

    # 等待逾時：至少 1 秒，或 3 個影格週期
    pop_timeout_us = int(max(1.0, 3.0 / max(args.fps, 0.1)) * 1e6)

    records = []
    copy_ms, debayer_ms = [], []
    n_pop_timeouts = 0
    n_bad_status = {}
    saved = 0
    prev_frame_id = None
    n_frame_id_gaps = 0

    cpu0 = os.times()
    t_start = time.monotonic()
    camera.start_acquisition()
    try:
        while time.monotonic() - t_start < args.duration:
            buf = stream.timeout_pop_buffer(pop_timeout_us)
            if buf is None:
                n_pop_timeouts += 1
                continue

            # 失敗的影格也更新 frame id，跳號只計算「完全沒收到」的影格
            frame_id = buf.get_frame_id()
            if prev_frame_id is not None and frame_id > prev_frame_id + 1:
                n_frame_id_gaps += frame_id - prev_frame_id - 1
            prev_frame_id = frame_id

            status = buf.get_status()
            if status != Aravis.BufferStatus.SUCCESS:
                name = status.value_nick if hasattr(status, 'value_nick') else str(status)
                n_bad_status[name] = n_bad_status.get(name, 0) + 1
                stream.push_buffer(buf)
                continue

            t0 = time.perf_counter()
            data = buf.get_data()                  # 複製一份，buffer 可以馬上還回串流
            width, height = buf.get_image_width(), buf.get_image_height()
            sys_ts_ns = buf.get_system_timestamp()
            stream.push_buffer(buf)
            t1 = time.perf_counter()
            copy_ms.append((t1 - t0) * 1000)

            dbr = None
            if not args.no_debayer or saved < args.save:
                img = to_bgr(data, width, height, pixel_format, args.cv_pattern)
                dbr = (time.perf_counter() - t1) * 1000
                debayer_ms.append(dbr)
                if saved < args.save:
                    cv2.imwrite(str(outdir / f'frame_{saved:02d}_id{frame_id}.jpg'), img,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1

            records.append({
                'frame_id': frame_id,
                'system_ts_ns': sys_ts_ns,
                'width': width,
                'height': height,
                'copy_ms': round((t1 - t0) * 1000, 3),
                'debayer_ms': None if dbr is None else round(dbr, 3),
            })
    except KeyboardInterrupt:
        print('\n中斷，整理結果中…')
    finally:
        camera.stop_acquisition()

    elapsed = time.monotonic() - t_start
    cpu1 = os.times()
    cpu_pct = ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / elapsed * 100
    stream_stats = read_stream_stats(stream)

    # 以系統時間戳記計算實際 fps（排除第一張）
    ts = [r['system_ts_ns'] for r in records]
    if len(ts) >= 2 and ts[-1] > ts[0]:
        fps_measured = (len(ts) - 1) / ((ts[-1] - ts[0]) / 1e9)
    else:
        fps_measured = len(records) / elapsed if elapsed > 0 else 0.0

    n_ok = len(records)
    n_bad = sum(n_bad_status.values())
    success_ratio = n_ok / (n_ok + n_bad) if (n_ok + n_bad) else 0.0
    per_frame_ms = copy_ms if args.no_debayer else [c + d for c, d in zip(copy_ms, debayer_ms)]
    proc_fps_limit = 1000 / statistics.mean(per_frame_ms) if per_frame_ms else 0.0
    passed = success_ratio >= 0.99 and fps_measured >= 0.9 * args.fps

    summary = {
        'time': datetime.now().isoformat(timespec='seconds'),
        'args': vars(args),
        'camera': info,
        'pixel_format': pixel_format,
        'receive_mode': ('packet_socket' if use_packet_socket else 'udp_socket') if is_gige else 'n/a',
        'config_notes': notes,
        'duration_s': round(elapsed, 2),
        'frames_ok': n_ok,
        'frames_bad_status': n_bad_status,
        'pop_timeouts': n_pop_timeouts,
        'frame_id_gaps': n_frame_id_gaps,
        'success_ratio': round(success_ratio, 4),
        'fps_target': args.fps,
        'fps_measured': round(fps_measured, 2),
        'throughput_mbps': round(fps_measured * payload * 8 / 1e6, 1),
        'cpu_percent_process': round(cpu_pct, 1),
        'copy_ms_mean': round(statistics.mean(copy_ms), 2) if copy_ms else None,
        'copy_ms_p95': round(pct(copy_ms, 95), 2) if copy_ms else None,
        'debayer_ms_mean': round(statistics.mean(debayer_ms), 2) if debayer_ms else None,
        'debayer_ms_p95': round(pct(debayer_ms, 95), 2) if debayer_ms else None,
        'single_thread_process_fps_limit': round(proc_fps_limit, 1),
        'stream_stats': stream_stats,
        'passed': passed,
    }

    with open(outdir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    if records:
        with open(outdir / 'frames.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    print('─' * 60)
    print(f"擷取時間      : {summary['duration_s']} s")
    print(f"成功 / 失敗   : {n_ok} / {n_bad} {n_bad_status or ''}（成功率 {success_ratio:.2%}）")
    print(f"實際 fps      : {summary['fps_measured']}（目標 {args.fps}）≈ {summary['throughput_mbps']} Mbps")
    print(f"frame id 跳號 : {n_frame_id_gaps}    pop 逾時：{n_pop_timeouts}")
    for key in ('n_received_packets', 'n_missing_packets', 'n_resent_packets', 'n_resend_requests',
                'n_missing_frames', 'n_underruns'):
        if key in stream_stats:
            print(f"{key:<14}: {stream_stats[key]}")
    print(f"CPU（本程序） : {summary['cpu_percent_process']}%（100% = 一個核心）")
    print(f"buffer 複製   : 平均 {summary['copy_ms_mean']} ms，p95 {summary['copy_ms_p95']} ms")
    if debayer_ms:
        print(f"解拜耳        : 平均 {summary['debayer_ms_mean']} ms，p95 {summary['debayer_ms_p95']} ms")
    print(f"單執行緒處理上限 ≈ {summary['single_thread_process_fps_limit']} fps（複製 + 解拜耳）")
    print(f"結果          : {'通過' if passed else '未達標'}")
    if n_ok == 0 and n_bad == 0:
        if stream_stats.get('n_received_packets', 0) == 0:
            print('[診斷] 完全沒收到影像封包：檢查防火牆（UDP）、相機與 Pi 是否同網段'
                  + ('；也可加 --no-packet-socket 比對 UDP 模式' if use_packet_socket else ''))
        else:
            print('[診斷] 有收到封包但沒有組成完整影格：檢查封包大小 / MTU，或降低 fps')
    print(f"輸出          : {outdir.resolve()}")
    if saved:
        print('請打開存下的 jpg 確認顏色：若紅藍對調或出現格紋，表示 Bayer 轉換碼不對，可用 --cv-pattern BG/RG/GB/GR 試換。')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
