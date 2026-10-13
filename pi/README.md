# Raspberry Pi 5 + GigE 相機擷取驗證

以 [Aravis](https://github.com/AravisProject/aravis) 0.8.36（穩定版）在 Raspberry Pi 5 上驗證 JHEM600GC-HM-P（3072×2048 彩色 GigE）的擷取能力。

| 檔案 | 用途 |
|---|---|
| `install_aravis.sh` | 安裝相依套件、編譯安裝 Aravis、設定 typelib / socket 緩衝 / cap_net_raw、建立 `~/aravis-venv` |
| `setup_gige_network.sh` | 相機網口固定 IP、自動測試 jumbo frame、加大 RX ring（NetworkManager） |
| `test_camera.sh` | 完整驗證流程，結果寫到 `output/pi_test_<時間>/` |
| `aravis_capture_test.py` | Python 擷取測試（fps、遺失封包、解拜耳耗時、CPU、存圖） |

## 前置條件

- Raspberry Pi OS 64-bit（Bookworm 或之後），5V/5A 電源，有散熱
- SSH 走 **Wi-Fi**，`eth0` 專門接相機（直連或經 PoE 注入器，不經交換器）
- Cat5e 以上網路線

## 步驟

```bash
# 0. 取得程式碼
git clone https://github.com/RobertLee0522/floor-water-drop.git ~/floor-water-drop
cd ~/floor-water-drop/pi

# 1. 安裝（約 5–10 分鐘）
bash install_aravis.sh
source /etc/profile.d/aravis.sh

# 2. 不接相機，先用模擬相機確認安裝正確
bash test_camera.sh --fake

# 3. 設定相機網口（相機 IP 需在同網段，例如 192.168.10.2）
bash setup_gige_network.sh --ip 192.168.10.1/24

# 4. 完整測試（約 3 分鐘）；--quick 只跑兩項，--udp 讓 Python 測試改用 UDP socket
bash test_camera.sh
```

`test_camera.sh` 結束代碼：0 = Python 擷取測試全部達標，1 = 有未達標，2 = 找不到相機 / 相機與 Pi 不同網段（會印出修正方式）。

## 看結果

`test_camera.sh` 最後會印出摘要表：

- **arv-camera-test 矩陣**：各 fps 的平均幀率、MiB/s、完成 / 失敗影格、遺失 / 重送封包
  - `f15_full` vs `f15_full_rt`：即時優先權有沒有幫助
  - `f10_full` vs `f10_full_udp`：packet socket 與一般 UDP socket 的差異
  - `f15_half_h`：相機端 ROI 減半高度的效果
- **Python 測試**：實際 fps、成功率、CPU、解拜耳耗時，以及 `python_f*/frame_*.jpg` 存圖

判讀重點：

| 現象 | 可能原因 / 對策 |
|---|---|
| 連線速度不是 1000 Mb/s | 網路線或 PoE 注入器問題 |
| `n_missing_packets` 多、失敗影格多 | 降 fps、開相機端 ROI、用 `--packet-delay` 拉開封包間隔、確認 packet socket 模式 |
| `n_underruns` > 0 | 處理太慢、緩衝不夠，加 `--buffers` 或降 fps |
| 影像紅藍對調 / 格紋 | Bayer 轉換碼不對，`aravis_capture_test.py --cv-pattern BG/RG/GB/GR` 試換 |
| 降頻狀態不是 `0x0` | 電源不足或過熱 |

## 單獨執行 Python 測試

```bash
~/aravis-venv/bin/python3 aravis_capture_test.py --probe
~/aravis-venv/bin/python3 aravis_capture_test.py --fps 15 --duration 30 --roi 0,512,3072,1024
~/aravis-venv/bin/python3 aravis_capture_test.py --help
```

`~/aravis-venv` 的直譯器是複製檔並帶有 `cap_net_raw`，Aravis 才能使用 packet socket（CPU 負擔較低）；系統的 `python3` 不受影響。
