#!/usr/bin/env bash
# =====================================================================
# Raspberry Pi 5 — GigE 相機專用網口設定（NetworkManager）
# =====================================================================
# 把相機直連的網口（預設 eth0）設成固定 IP，並嘗試開啟 jumbo frame、
# 加大網卡 RX ring。設定寫入 NetworkManager 連線 "gige-camera"，重開機後保留。
#
# 用法：
#   bash setup_gige_network.sh                          # eth0, 192.168.10.1/24, MTU 自動
#   bash setup_gige_network.sh --ip 192.168.1.10/24     # 配合相機現有網段
#   bash setup_gige_network.sh --mtu 1500               # 固定 MTU
#   bash setup_gige_network.sh --dry-run                # 只顯示將執行的動作
#
# 注意：若目前是透過同一個網口 SSH 進來，設定後連線會中斷。
#       建議 SSH 走 Wi-Fi，eth0 專給相機使用。
# =====================================================================
set -euo pipefail

IFACE="eth0"
ADDR="192.168.10.1/24"
MTU="auto"
CON_NAME="gige-camera"
DRY_RUN=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface)   IFACE="$2"; shift 2 ;;
        --ip)      ADDR="$2"; shift 2 ;;
        --mtu)     MTU="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --force)   FORCE=1; shift ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知參數：$1"; exit 1 ;;
    esac
done

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[警告] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[錯誤] %s\033[0m\n' "$*"; exit 1; }
run()  { echo "+ $*"; [[ $DRY_RUN -eq 1 ]] || "$@"; }

[[ "$ADDR" == */* ]] || die "--ip 需包含遮罩，例如 192.168.10.1/24"
ip link show "$IFACE" >/dev/null 2>&1 || die "找不到網口 $IFACE"
command -v nmcli >/dev/null || die "找不到 nmcli；此腳本適用 NetworkManager（Raspberry Pi OS Bookworm 之後的預設）"
systemctl is-active --quiet NetworkManager || die "NetworkManager 未執行"

# ─────────── SSH 防呆 ───────────
if [[ -n "${SSH_CONNECTION:-}" && $FORCE -eq 0 ]]; then
    SSH_LOCAL_IP="$(awk '{print $3}' <<<"$SSH_CONNECTION")"
    IFACE_ADDRS="$(ip -4 -br addr show dev "$IFACE")"
    if [[ "$IFACE_ADDRS" == *" ${SSH_LOCAL_IP}/"* ]]; then
        die "目前 SSH 連線走 ${IFACE}（${SSH_LOCAL_IP}），修改後會斷線。請改用 Wi-Fi 連線，或加 --force"
    fi
fi

log "目前狀態"
ip -br addr show dev "$IFACE"
ip -d link show "$IFACE" | grep -oE 'mtu [0-9]+|maxmtu [0-9]+' | tr '\n' ' '; echo
sudo ethtool "$IFACE" 2>/dev/null | grep -E 'Speed|Duplex|Link detected' || true

# ─────────── MTU：由大到小嘗試 ───────────
if [[ "$MTU" == "auto" ]]; then
    log "測試網卡可接受的最大 MTU"
    MAXMTU="$(ip -d link show "$IFACE" | grep -oE 'maxmtu [0-9]+' | awk '{print $2}')"
    [[ -n "$MAXMTU" ]] && echo "驅動回報 maxmtu = ${MAXMTU}"
    ORIG_MTU="$(cat "/sys/class/net/${IFACE}/mtu")"
    MTU=1500
    if [[ $DRY_RUN -eq 0 ]]; then
        for m in 9000 8192 7200 4096 3000 1500; do
            [[ -n "$MAXMTU" && $m -gt $MAXMTU ]] && continue
            if sudo ip link set dev "$IFACE" mtu "$m" 2>/dev/null; then MTU=$m; break; fi
        done
        sudo ip link set dev "$IFACE" mtu "$ORIG_MTU"
    fi
    echo "採用 MTU = ${MTU}"
    [[ $MTU -le 1500 ]] && warn "網卡不支援 jumbo frame；相機封包大小會維持 1500 左右，CPU 負擔較高"
fi

# ─────────── RX ring ───────────
RX_MAX=""
if RING="$(sudo ethtool -g "$IFACE" 2>/dev/null)"; then
    RX_MAX="$(awk '/Pre-set maximums/{f=1} f && /^RX:/{print $2; exit}' <<<"$RING")"
    [[ "$RX_MAX" =~ ^[0-9]+$ ]] && echo "RX ring 上限 = ${RX_MAX}" || RX_MAX=""
fi
[[ -z "$RX_MAX" ]] && echo "網卡不支援調整 RX ring，略過"

# ─────────── NetworkManager 連線 ───────────
log "寫入 NetworkManager 連線 ${CON_NAME}"
PROPS=(
    connection.interface-name "$IFACE"
    connection.autoconnect yes
    connection.autoconnect-priority 100
    ipv4.method manual
    ipv4.addresses "$ADDR"
    ipv4.never-default yes          # 不搶預設路由，上網仍走 Wi-Fi
    ipv6.method disabled
    802-3-ethernet.mtu "$MTU"
)
[[ -n "$RX_MAX" ]] && PROPS+=(ethtool.ring-rx "$RX_MAX")

if nmcli connection show "$CON_NAME" >/dev/null 2>&1; then
    run sudo nmcli connection modify "$CON_NAME" "${PROPS[@]}"
else
    run sudo nmcli connection add type ethernet con-name "$CON_NAME" "${PROPS[@]}"
fi
run sudo nmcli connection up "$CON_NAME"

log "設定後狀態"
ip -br addr show dev "$IFACE"
echo "MTU = $(cat "/sys/class/net/${IFACE}/mtu")"

cat <<EOF

相機 IP 需與 ${ADDR%/*} 同網段（例如 ${ADDR%.*}.2）。
若相機目前在其他網段，可先暫時加一個同網段位址讓 Pi 連得到相機，再改相機的 persistent IP：
  sudo ip addr add <相機網段>.250/24 dev ${IFACE}
  arv-tool-0.8 -n "<相機名稱>" network mode=PersistentIP ip=${ADDR%.*}.2 mask=255.255.255.0
  （相機重新上電後生效，再用 sudo ip addr del ... 移除暫時位址）
不知道相機目前 IP 時，test_camera.sh 找不到相機會自動用 tcpdump 監聽 GVCP 封包。
EOF
