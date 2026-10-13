# Raspberry Pi 5 部署 Prompt

先 `git clone https://github.com/RobertLee0522/floor-water-drop.git ~/floor-water-drop`，在 `~/floor-water-drop/pi` 目錄啟動 Claude Code，貼上以下內容。測試輸出會寫到 `~/floor-water-drop/output/`（已在 .gitignore）。

---

你在一台 Raspberry Pi 5 上，任務是部署並驗證 GigE 工業相機的影像擷取，作為地板水滴偵測專案的前端。請依下列步驟執行，每一步都要看實際輸出再往下，不要預測結果。

## 背景

- 相機：JHEM600GC-HM-P（GigE，可能是 PoE），彩色，3072×2048（約 6.3 MP）。Bayer8 在 16 fps 約 800 Mbps，接近 Gigabit 上限，頻寬是最大風險。
- 函式庫：Aravis 0.8.36（GenICam / GigE Vision），從原始碼編譯到 `/usr/local`。
- 目前目錄的腳本（已在 x86 WSL 用 Aravis 模擬 GigE 相機驗證過，尚未在 Pi 與真相機上跑過）：
  - `install_aravis.sh`：安裝相依、編譯 Aravis、設定 typelib / sysctl / cap_net_raw、建立 `~/aravis-venv`
  - `setup_gige_network.sh`：eth0 固定 IP、自動測 jumbo frame、RX ring（NetworkManager 連線 `gige-camera`）
  - `test_camera.sh`：系統檢查 → 找相機 → 相機資訊 → arv-camera-test 頻寬矩陣 → Python 擷取測試。結束代碼 0 = 達標、1 = 未達標、2 = 找不到相機或不同網段
  - `aravis_capture_test.py`：Python 擷取測試（`--help` 看參數；`--fake` 用模擬相機；`--cv-pattern` 修正 Bayer 顏色）
  - `README.md`：使用說明與結果判讀表

## 規則

- **網路安全**：修改 eth0 前先確認我的 SSH 不是走 eth0（看 `$SSH_CONNECTION` 與 `ip -br addr`）。如果是，停下來告訴我，不要執行 `setup_gige_network.sh --force`。
- **需要我同意才做**：執行 `setup_gige_network.sh`（會改 NetworkManager 設定）、修改相機的 persistent IP（`arv-tool-0.8 ... network ...`）、刪除任何檔案。先說明要執行的指令與影響，等我回覆。
- **腳本有 bug**：先找出根因，做最小修改，並在報告中列出 diff 與原因。不要為了讓測試通過而放寬判斷標準。
- 所有輸出都保留在 `../output/` 下，不要刪除舊的測試結果。

## 步驟

### 0. 前置檢查
- `uname -m`（應為 aarch64）、`cat /proc/device-tree/model`、`cat /etc/os-release`、`df -h ~`、`free -h`
- 確認可以連網（`curl -sI https://github.com`）
- `file *.sh *.py` 確認沒有 CRLF；若有，用 `sed -i 's/\r$//' *.sh *.py` 修正並告訴我
- `vcgencmd get_throttled`（應為 `throttled=0x0`，否則先提醒我電源或散熱問題）

### 1. 安裝
- `bash install_aravis.sh 2>&1 | tee ../output/install.log`
- 失敗時看 log 找根因。已知風險：較新的 Raspberry Pi OS（Debian Trixie）套件名稱可能不同（例如 `libgirepository1.0-dev`），meson 找不到 `gobject-introspection-1.0` 時要找替代套件。
- 成功標準：最後印出 `PyGObject Aravis-0.8 OK ...`
- 之後執行 `source /etc/profile.d/aravis.sh`

### 2. 模擬相機驗證
- `bash test_camera.sh --fake -d 3`，期望結束代碼 0、兩項 Python 測試都「通過」

### 3. 相機連線與網路
- 確認相機接在 eth0：`ip -br link show eth0`、`cat /sys/class/net/eth0/speed`（應為 1000）
- `arv-tool-0.8` 列出相機。找不到時，`sudo timeout 8 tcpdump -n -i eth0 udp port 3956` 同時在另一個指令執行 `arv-tool-0.8`，從回應封包找出相機目前的 IP
- 依相機目前的 IP 提出網路方案（Pi 配合相機網段，或之後改相機 IP），**等我同意**後執行 `bash setup_gige_network.sh --ip <Pi IP>/24`
- 記錄 `setup_gige_network.sh` 選到的 MTU：**Pi 5 網卡是否支援 jumbo frame 是重點未知數**

### 4. 完整測試
- `bash test_camera.sh 2>&1 | tail -60`，記下輸出資料夾
- 結束代碼 2：照輸出的提示處理（不同網段的修正方式會印出來）
- 頻寬矩陣有失敗影格或遺失封包時，參考 README 的判讀表

### 5. 確認影像顏色
- 用 Read 工具打開 `../output/pi_test_*/python_f10/frame_00_*.jpg`，看顏色是否自然（地板、牆面不應偏紅 / 偏藍，不應有格紋）
- 若顏色不對，用 `~/aravis-venv/bin/python3 aravis_capture_test.py --fps 5 --duration 3 --save 1 --cv-pattern <BG|RG|GB|GR>` 逐一試，找出正確的那個

### 6. 找出可穩定運作的設定
用 `aravis_capture_test.py` 搜尋以下兩種設定的最高穩定 fps（條件：持續 30 秒、成功率 100%、`n_missing_packets` 為 0 或接近 0）：
- 全幅 3072×2048
- 地板 ROI（先用 `--roi 0,512,3072,1024` 做代表）

可調整的參數：`--fps`、`--packet-size`（MTU 有放大時）、`--packet-delay`（遺失封包多時，從 1000 ns 起試）、`--buffers`。同時比較 packet socket 與 `--no-packet-socket` 的 CPU 使用率。

### 7. 報告
把結果寫到 `../output/DEPLOY_REPORT.md`，內容包括：
1. 環境：機型、OS、kernel、Aravis 版本、相機型號 / 韌體 / 序號 / IP、像素格式
2. 網路：連線速度、採用的 MTU、jumbo frame 是否支援、RX ring 是否可調
3. `test_camera.sh` 摘要表（原樣貼上）
4. 第 6 步的結果表：設定 → 實際 fps、成功率、遺失封包、CPU、解拜耳耗時
5. 正確的 Bayer 轉換（`--cv-pattern` 值，或預設即正確）
6. 測試前後的溫度與降頻狀態
7. 建議：給水滴偵測使用的 fps、ROI、封包大小、接收模式，以及原因
8. 修改過的腳本：diff 與原因（沒有修改就寫「無」）

最後告訴我報告路徑，並列出我要帶回 Windows 的檔案：`DEPLOY_REPORT.md`、`test.log`、`summary.tsv`、各 `summary.json`、每種設定一張樣本 jpg。
