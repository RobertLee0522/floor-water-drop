#!/usr/bin/env bash
# =====================================================================
# Raspberry Pi 5 — GigE 相機擷取驗證
# =====================================================================
# 步驟：
#   1. 系統檢查：Aravis 版本、網口速度 / MTU、socket 緩衝、權限、溫度與降頻
#   2. 搜尋相機（找不到時自動用 tcpdump 監聽 GVCP，找出相機實際 IP）
#   3. 讀取相機資訊：網路設定、GenICam XML、所有特徵值、建議像素格式
#   4. arv-camera-test 頻寬矩陣：不同 fps / 接收模式 / ROI 的完成率與遺失封包
#   5. Python 擷取測試：實際 fps、複製與解拜耳耗時、CPU 使用率、存圖確認顏色
# 所有輸出寫到 output/pi_test_<時間>/，最後印出摘要表。
#
# 用法：
#   bash test_camera.sh                      # 自動選第一台相機
#   bash test_camera.sh -n "<相機名稱>"       # 指定相機
#   bash test_camera.sh -d 20                # 每項測試 20 秒（預設 10）
#   bash test_camera.sh --quick              # 只跑 10 fps 與最高速兩項
#   bash test_camera.sh --fake               # 不接相機，用模擬相機驗證安裝
#   bash test_camera.sh --udp                # Python 測試改用 UDP socket（packet socket 收不到時比對用）
#
# 結束代碼：0 = Python 擷取測試全部達標，1 = 有未達標項目，2 = 找不到 / 無法開啟相機
# =====================================================================
set -uo pipefail

API="0.8"
CAM=""
IFACE="eth0"
DURATION=10
QUICK=0
FAKE=0
OUT=""
PY_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--name)     CAM="$2"; shift 2 ;;
        -i|--iface)    IFACE="$2"; shift 2 ;;
        -d|--duration) DURATION="$2"; shift 2 ;;
        -o|--outdir)   OUT="$2"; shift 2 ;;
        --quick)       QUICK=1; shift ;;
        --fake)        FAKE=1; shift ;;
        --udp)         PY_EXTRA=(--no-packet-socket); shift ;;
        -h|--help)     sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知參數：$1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[[ -f /etc/profile.d/aravis.sh ]] && . /etc/profile.d/aravis.sh
ARV_TOOL="arv-tool-${API}"
ARV_TEST="arv-camera-test-${API}"
PY="${ARAVIS_PY:-$HOME/aravis-venv/bin/python3}"
[[ -x "$PY" ]] || PY="python3"
CAPTURE="${SCRIPT_DIR}/aravis_capture_test.py"
OUT="${OUT:-${SCRIPT_DIR}/../output/pi_test_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
exec > >(tee -a "${OUT}/test.log") 2>&1

section() { printf '\n\033[1;36m━━━ %s ━━━\033[0m\n' "$*"; }
warn()    { printf '\033[1;33m[警告] %s\033[0m\n' "$*"; WARNINGS+=("$*"); }
die()     { printf '\033[1;31m[錯誤] %s\033[0m\n' "$*"; exit 2; }
WARNINGS=()

command -v "$ARV_TOOL" >/dev/null || die "找不到 ${ARV_TOOL}，請先執行 install_aravis.sh"

# ─────────── 1. 系統檢查 ───────────
section "1. 系統檢查"
echo "機型      : $(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"
echo "系統      : $(. /etc/os-release; echo "$PRETTY_NAME") / kernel $(uname -r)"
echo "Aravis    : $("$ARV_TOOL" --version 2>&1 | head -1)"
echo "Python    : ${PY}"
echo "CPU 調速  : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
echo "rmem_max  : $(sysctl -n net.core.rmem_max)"
CAP_TEST="$(getcap "$(command -v "$ARV_TEST")" 2>/dev/null)"
CAP_PY="$(getcap "$(readlink -f "$PY")" 2>/dev/null)"
echo "權限      : ${CAP_TEST:-arv-camera-test 無 cap_net_raw} | ${CAP_PY:-Python 無 cap_net_raw}"
[[ "$CAP_TEST" == *cap_net_raw* ]] || warn "arv-camera-test 沒有 cap_net_raw，會退回 UDP socket 模式"
[[ "$CAP_PY" == *cap_net_raw* ]]   || warn "Python 直譯器沒有 cap_net_raw，Python 測試會使用 UDP socket 模式"

if command -v vcgencmd >/dev/null; then
    echo "溫度      : $(vcgencmd measure_temp)"
    THROTTLED_BEFORE="$(vcgencmd get_throttled)"
    echo "降頻狀態  : ${THROTTLED_BEFORE}"
    [[ "$THROTTLED_BEFORE" == "throttled=0x0" ]] || warn "偵測到降頻 / 電壓不足（${THROTTLED_BEFORE}），請確認使用 5V/5A 電源與散熱"
fi

if [[ $FAKE -eq 0 ]]; then
    echo
    ip -br addr show dev "$IFACE" || die "找不到網口 ${IFACE}"
    MTU_NOW="$(cat "/sys/class/net/${IFACE}/mtu")"
    SPEED="$(cat "/sys/class/net/${IFACE}/speed" 2>/dev/null || echo "?")"
    echo "MTU       : ${MTU_NOW}"
    echo "連線速度  : ${SPEED} Mb/s"
    if [[ ! "$SPEED" =~ ^[0-9]+$ || $SPEED -lt 1000 ]]; then
        warn "${IFACE} 連線速度為 ${SPEED} Mb/s（應為 1000），請檢查網路線（需 Cat5e 以上）與 PoE 注入器"
    fi
    [[ $MTU_NOW -gt 1500 ]]  || warn "MTU ${MTU_NOW}，未啟用 jumbo frame（可執行 setup_gige_network.sh 嘗試）"
fi

# ─────────── 2. 搜尋相機 ───────────
ip2int() { local IFS=.; local a b c d; read -r a b c d <<<"$1"; echo $(( (a << 24) | (b << 16) | (c << 8) | d )); }

# 相機 IP 是否落在網口任一 IPv4 位址的網段內
in_iface_subnet() {
    local cam_ip="$1" cidr net pfx mask
    for cidr in $(ip -4 -o addr show dev "$IFACE" | awk '{print $4}'); do
        net="${cidr%/*}"; pfx="${cidr#*/}"
        mask=$(( pfx == 0 ? 0 : (0xFFFFFFFF << (32 - pfx)) & 0xFFFFFFFF ))
        (( ($(ip2int "$net") & mask) == ($(ip2int "$cam_ip") & mask) )) && return 0
    done
    return 1
}

subnet_hint() {
    local cam_ip="$1"
    echo "相機 IP ${cam_ip} 與 Pi 的 ${IFACE}（$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | tr '\n' ' ')）不同網段，Aravis 能找到但無法控制。處理方式："
    echo "  1. 暫時加同網段位址：sudo ip addr add ${cam_ip%.*}.250/24 dev ${IFACE}"
    echo "  2. 改相機 IP：${ARV_TOOL} -n \"<相機名稱>\" network mode=PersistentIP ip=<Pi 網段的新 IP> mask=255.255.255.0"
    echo "     （相機重新上電後生效，再用 sudo ip addr del ${cam_ip%.*}.250/24 dev ${IFACE} 移除暫時位址）"
    echo "  或執行 setup_gige_network.sh --ip ${cam_ip%.*}.1/24，讓 Pi 配合相機網段"
}

section "2. 搜尋相機"
if [[ $FAKE -eq 1 ]]; then
    CAM="Fake_1"
    echo "使用模擬相機 Fake_1（arv-camera-test 測試略過，只跑 Python 測試）"
else
    "$ARV_TOOL" > "${OUT}/devices.txt" 2>&1
    cat "${OUT}/devices.txt"
    if ! grep -q " (" "${OUT}/devices.txt"; then
        echo
        echo "找不到相機，監聽 ${IFACE} 上的 GVCP（UDP 3956）封包 6 秒，找出相機實際 IP…"
        sudo timeout 6 tcpdump -n -l -i "$IFACE" udp port 3956 > "${OUT}/gvcp_tcpdump.txt" 2>&1 &
        TCPDUMP_PID=$!
        sleep 1
        "$ARV_TOOL" >/dev/null 2>&1
        sleep 1
        "$ARV_TOOL" >/dev/null 2>&1
        wait "$TCPDUMP_PID"
        grep -E "IP " "${OUT}/gvcp_tcpdump.txt" | head -10
        CAM_IPS="$(grep -oE 'IP [0-9.]+\.3956 >' "${OUT}/gvcp_tcpdump.txt" | awk '{print $2}' | sed 's/\.3956$//' | sort -u)"
        echo
        if [[ -n "$CAM_IPS" ]]; then
            echo "相機有回應，IP：${CAM_IPS}"
            subnet_hint "$(head -1 <<<"$CAM_IPS")"
        else
            echo "沒有收到任何 GVCP 回應。請確認："
            echo "  - 相機有供電（PoE 注入器燈號）、網路線接在 ${IFACE}"
            echo "  - ip link 顯示 ${IFACE} 為 UP，連線速度 1000 Mb/s"
            echo "  - 相機剛上電需等約 20–60 秒（DHCP 逾時後才會退到 LLA 169.254.x.x）"
        fi
        die "找不到相機"
    fi
    if [[ -z "$CAM" ]]; then
        CAM="$(grep " (" "${OUT}/devices.txt" | head -1 | sed 's/ (.*//')"
        [[ $(grep -c " (" "${OUT}/devices.txt") -gt 1 ]] && warn "找到多台相機，使用第一台：${CAM}（可用 -n 指定）"
    fi
    CAM_IP="$(grep -F "${CAM} (" "${OUT}/devices.txt" | head -1 | sed -n 's/.*(\([0-9.]*\)).*/\1/p')"
    if [[ -n "$CAM_IP" ]] && ! in_iface_subnet "$CAM_IP"; then
        echo
        subnet_hint "$CAM_IP"
        die "相機與 ${IFACE} 不同網段"
    fi
fi
echo "測試相機：${CAM}"

# ─────────── 3. 相機資訊 ───────────
section "3. 相機資訊"
if [[ $FAKE -eq 0 ]]; then
    "$ARV_TOOL" -n "$CAM" network 2>&1 | tee "${OUT}/network.txt"
    "$ARV_TOOL" -n "$CAM" genicam > "${OUT}/genicam.xml" 2>&1
    echo "（讀取所有特徵值中，可能需要數十秒…）"
    "$ARV_TOOL" -n "$CAM" values > "${OUT}/values.txt" 2>&1
    grep -E "^\s*(Enumeration|Integer|Float|String|Boolean).*'(DeviceVendorName|DeviceModelName|DeviceFirmwareVersion|PixelFormat|Width|Height|WidthMax|HeightMax|AcquisitionFrameRate|AcquisitionFrameRateEnable|ExposureTime|ExposureAuto|GevSCPSPacketSize|GevSCPD|DeviceLinkThroughputLimit)'" \
        "${OUT}/values.txt" | head -30
fi

FAKE_ARG=()
[[ $FAKE -eq 1 ]] && FAKE_ARG=(--fake)
"$PY" "$CAPTURE" "${FAKE_ARG[@]}" --camera "$CAM" --probe > "${OUT}/probe.txt" 2>&1
PROBE_RC=$?
sed '/^BEST_FORMAT=/d; /^SENSOR_SIZE=/d' "${OUT}/probe.txt"
[[ $PROBE_RC -eq 0 ]] || die "Python 無法開啟相機，請看 ${OUT}/probe.txt"
FMT="$(sed -n 's/^BEST_FORMAT=//p' "${OUT}/probe.txt")"
echo "建議像素格式：${FMT:-（無 Bayer8/Mono8，沿用目前格式）}"
SENSOR="$(sed -n 's/^SENSOR_SIZE=//p' "${OUT}/probe.txt")"
SW="${SENSOR%x*}"; SH="${SENSOR#*x}"
echo "感測器尺寸：${SENSOR:-未知}"
FULL=(); HALF=()
if [[ "$SW" =~ ^[0-9]+$ && "$SH" =~ ^[0-9]+$ ]]; then
    FULL=(-w "$SW" -h "$SH")
    HALF=(-w "$SW" -h $((SH / 2)))
else
    warn "無法取得感測器尺寸，頻寬矩陣沿用相機目前的 ROI"
fi

# ─────────── 4. arv-camera-test 頻寬矩陣 ───────────
SUMMARY="${OUT}/summary.tsv"
printf 'test\tfps\tMiB/s\tcompleted\tfailures\tunderruns\tmissing_pkt\tresent_pkt\n' > "$SUMMARY"

run_bench() {
    local label="$1"; shift
    local log="${OUT}/bench_${label}.log"
    local feat=()
    [[ -n "$FMT" ]] && feat=(--features "PixelFormat=${FMT}")
    echo
    echo "▶ ${label}：${ARV_TEST} $* （${DURATION} 秒）"
    timeout $((DURATION + 30)) "$ARV_TEST" -n "$CAM" --auto --duration "$DURATION" "${feat[@]}" "$@" > "$log" 2>&1
    # 每秒輸出格式：" 11 frames/s -    2.88 MiB/s"，第一秒為暖機不列入平均
    local fps mibs
    fps="$(awk '/frames?\/s/ && /MiB\/s/{n++; if(n>1){s+=$1; c++}} END{if(c) printf "%.1f", s/c; else print "-"}' "$log")"
    mibs="$(awk '/frames?\/s/ && /MiB\/s/{n++; if(n>1){s+=$4; c++}} END{if(c) printf "%.1f", s/c; else print "-"}' "$log")"
    getstat() { awk -v k="$1" '$1==k {print $3}' "$log" | tail -1; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$fps" "$mibs" \
        "$(getstat n_completed_buffers)" "$(getstat n_failures)" "$(getstat n_underruns)" \
        "$(getstat n_missing_packets)" "$(getstat n_resent_packets)" | tee -a "$SUMMARY"
}

if [[ $FAKE -eq 0 ]]; then
    section "4. arv-camera-test 頻寬矩陣"
    echo "欄位：平均 fps、MiB/s、完成影格、失敗影格、緩衝不足、遺失封包、重送封包"
    if [[ $QUICK -eq 1 ]]; then
        run_bench f10_full      "${FULL[@]}" -f 10
        run_bench fmax_full     "${FULL[@]}" -f 1000
    else
        run_bench f05_full      "${FULL[@]}" -f 5
        run_bench f10_full      "${FULL[@]}" -f 10
        run_bench f15_full      "${FULL[@]}" -f 15
        # -f 1000：Aravis 會自動夾到相機在此 ROI 下的最高幀率
        run_bench fmax_full     "${FULL[@]}" -f 1000
        run_bench f15_full_rt   "${FULL[@]}" -f 15 --realtime
        run_bench f10_full_udp  "${FULL[@]}" -f 10 --no-packet-socket
        # ROI 測試放最後：arv-camera-test 改的 ROI 會保留在相機上，第 5 步會恢復全幅
        [[ ${#HALF[@]} -gt 0 ]] && run_bench f15_half_h "${HALF[@]}" -f 15
    fi
fi

# ─────────── 5. Python 擷取測試 ───────────
section "5. Python 擷取測試（全幅，含解拜耳）"
PY_RESULTS=()
PY_FAILED=0
for fps in 5 10; do
    [[ $QUICK -eq 1 && $fps -eq 5 ]] && continue
    echo
    echo "▶ python_f${fps}"
    "$PY" "$CAPTURE" "${FAKE_ARG[@]}" "${PY_EXTRA[@]}" --camera "$CAM" --roi full --fps "$fps" \
        --duration $((DURATION * 2)) --save 3 --outdir "${OUT}/python_f${fps}"
    rc=$?
    [[ $rc -eq 0 ]] || PY_FAILED=1
    PY_RESULTS+=("python_f${fps}: exit ${rc}")
done

# ─────────── 摘要 ───────────
section "摘要"
echo "相機：${CAM}    像素格式：${FMT:-?}    網口 MTU：${MTU_NOW:-n/a}"
if [[ $FAKE -eq 0 ]]; then
    echo
    column -t -s $'\t' "$SUMMARY"
fi
echo
for r in "${PY_RESULTS[@]}"; do
    d="${OUT}/${r%%:*}"
    if [[ -f "$d/summary.json" ]]; then
        "$PY" - "$d/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding='utf-8'))
st = s.get('stream_stats', {})
print(f"{s['args']['fps']:>4.0f} fps 目標 → 實際 {s['fps_measured']} fps | 成功率 {s['success_ratio']:.2%} | "
      f"遺失封包 {st.get('n_missing_packets', '-')} | CPU {s['cpu_percent_process']}% | "
      f"解拜耳 {s['debayer_ms_mean']} ms | {s['receive_mode']} | {'通過' if s['passed'] else '未達標'}")
EOF
    else
        echo "$r（無結果檔）"
    fi
done

if command -v vcgencmd >/dev/null; then
    echo
    echo "測試後溫度：$(vcgencmd measure_temp)    降頻狀態：$(vcgencmd get_throttled)"
fi

if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    echo
    echo "警告："
    printf '  - %s\n' "${WARNINGS[@]}"
fi

echo
echo "完整紀錄：${OUT}"
echo "請打開 ${OUT}/python_f*/frame_*.jpg 確認影像顏色正確。"
exit "$PY_FAILED"
