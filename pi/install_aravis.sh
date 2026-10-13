#!/usr/bin/env bash
# =====================================================================
# Raspberry Pi 5 — Aravis（GigE Vision）安裝腳本
# =====================================================================
# 從官方 release tarball 編譯 Aravis 穩定版（預設 0.8.36），只啟用 GigE
# 相關功能，並完成以下設定：
#   1. 安裝編譯與 Python 相依套件（apt）
#   2. 下載、驗證 SHA256、編譯、安裝到 /usr/local
#   3. /etc/profile.d/aravis.sh 設定 GI_TYPELIB_PATH
#   4. /etc/sysctl.d/60-aravis-gige.conf 加大 socket 接收緩衝上限
#   5. arv-camera-test 加上 cap_net_raw（啟用 packet socket 接收模式）
#   6. 建立 ~/aravis-venv（複製版直譯器 + cap_net_raw），給 Python 測試用
#
# 用法：
#   bash install_aravis.sh                 # 使用預設版本
#   bash install_aravis.sh --version 0.8.35 --sha256 <hash>
#   bash install_aravis.sh --no-venv       # 不建立帶 cap_net_raw 的 venv
#
# 以一般使用者執行，需要 root 的步驟會自動使用 sudo。
# =====================================================================
set -euo pipefail

ARAVIS_VERSION="0.8.36"
ARAVIS_SHA256="246deaa0042a387ff1bc00332d0fb80537ce14abde2c28d1a54c91f17adc51bf"
PREFIX="/usr/local"
BUILD_ROOT="${HOME}/aravis-build"
VENV_DIR="${HOME}/aravis-venv"
MAKE_VENV=1
APPLY_SYSCTL=1
SKIP_VERIFY=0

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)      ARAVIS_VERSION="$2"; ARAVIS_SHA256=""; shift 2 ;;
        --sha256)       ARAVIS_SHA256="$2"; shift 2 ;;
        --skip-verify)  SKIP_VERIFY=1; shift ;;
        --prefix)       PREFIX="$2"; shift 2 ;;
        --venv)         VENV_DIR="$2"; shift 2 ;;
        --no-venv)      MAKE_VENV=0; shift ;;
        --no-sysctl)    APPLY_SYSCTL=0; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "未知參數：$1"; usage 1 ;;
    esac
done

API="${ARAVIS_VERSION%.*}"          # 0.8.36 → 0.8
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[警告] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[錯誤] %s\033[0m\n' "$*"; exit 1; }

# ─────────── 0. 環境檢查 ───────────
log "環境檢查"
[[ $EUID -eq 0 ]] && die "請用一般使用者執行（腳本內會自行 sudo）"
[[ "$(uname -m)" == "aarch64" ]] || warn "目前架構為 $(uname -m)，預期為 aarch64（64-bit Raspberry Pi OS）"
MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
echo "機型：${MODEL}"
[[ "$MODEL" == *"Raspberry Pi 5"* ]] || warn "非 Raspberry Pi 5，腳本仍會繼續"
. /etc/os-release && echo "系統：${PRETTY_NAME}"
if [[ -z "$ARAVIS_SHA256" && $SKIP_VERIFY -eq 0 ]]; then
    die "指定了 --version 時請一併給 --sha256，或加上 --skip-verify"
fi

# ─────────── 1. apt 相依套件 ───────────
log "安裝相依套件"
sudo apt-get update
PKGS=(
    build-essential meson ninja-build pkg-config curl xz-utils
    libglib2.0-dev libxml2-dev zlib1g-dev
    gobject-introspection
    python3-gi python3-numpy python3-opencv python3-venv
    libcap2-bin ethtool tcpdump
)
# libgirepository1.0-dev 提供 gobject-introspection-1.0.pc（編譯 typelib 用）
if apt-cache show libgirepository1.0-dev >/dev/null 2>&1; then
    PKGS+=(libgirepository1.0-dev)
else
    warn "找不到 libgirepository1.0-dev，若 meson 找不到 gobject-introspection-1.0 請手動安裝對應套件"
fi
sudo apt-get install -y "${PKGS[@]}"

if dpkg -s "gir1.2-aravis-${API}" >/dev/null 2>&1; then
    warn "系統已裝 apt 版 gir1.2-aravis-${API}；GI_TYPELIB_PATH 會讓 /usr/local 版本優先"
fi

# ─────────── 2. 下載、驗證、編譯 ───────────
log "下載 Aravis ${ARAVIS_VERSION}"
mkdir -p "$BUILD_ROOT"
TARBALL="${BUILD_ROOT}/aravis-${ARAVIS_VERSION}.tar.xz"
SRC_DIR="${BUILD_ROOT}/aravis-${ARAVIS_VERSION}"
BUILD_DIR="${SRC_DIR}/build"
URL="https://github.com/AravisProject/aravis/releases/download/${ARAVIS_VERSION}/aravis-${ARAVIS_VERSION}.tar.xz"

[[ -f "$TARBALL" ]] || curl -fL --retry 3 -o "$TARBALL" "$URL"
if [[ $SKIP_VERIFY -eq 0 ]]; then
    echo "${ARAVIS_SHA256}  ${TARBALL}" | sha256sum -c - || die "SHA256 不符，已中止"
fi
rm -rf "$SRC_DIR"
tar -xJf "$TARBALL" -C "$BUILD_ROOT"

log "meson 設定（僅 GigE + packet socket + introspection）"
meson setup "$BUILD_DIR" "$SRC_DIR" \
    --prefix="$PREFIX" \
    --buildtype=release \
    -Dviewer=disabled \
    -Dgst-plugin=disabled \
    -Dusb=disabled \
    -Dpacket-socket=enabled \
    -Dintrospection=enabled \
    -Dtests=false \
    -Ddocumentation=disabled

log "編譯（Pi 5 約需數分鐘）"
ninja -C "$BUILD_DIR"

log "安裝到 ${PREFIX}"
sudo ninja -C "$BUILD_DIR" install
sudo ldconfig

# ─────────── 3. GI_TYPELIB_PATH ───────────
log "設定 GI_TYPELIB_PATH"
TYPELIB_FILE="$(find "$PREFIX/lib" -name "Aravis-${API}.typelib" -print -quit)"
[[ -n "$TYPELIB_FILE" ]] || die "找不到 Aravis-${API}.typelib，introspection 可能沒有編譯成功"
TYPELIB_DIR="$(dirname "$TYPELIB_FILE")"
echo "typelib：${TYPELIB_DIR}"
sudo tee /etc/profile.d/aravis.sh >/dev/null <<EOF
# 由 install_aravis.sh 產生：讓 PyGObject 找到 /usr/local 下的 Aravis typelib
export GI_TYPELIB_PATH="${TYPELIB_DIR}\${GI_TYPELIB_PATH:+:\$GI_TYPELIB_PATH}"
EOF
export GI_TYPELIB_PATH="${TYPELIB_DIR}${GI_TYPELIB_PATH:+:$GI_TYPELIB_PATH}"

# ─────────── 4. socket 緩衝上限 ───────────
if [[ $APPLY_SYSCTL -eq 1 ]]; then
    log "加大 socket 接收緩衝上限（UDP 模式時 socket-buffer=AUTO 需要）"
    sudo tee /etc/sysctl.d/60-aravis-gige.conf >/dev/null <<'EOF'
# 由 install_aravis.sh 產生：GigE Vision 影像串流接收緩衝
# 6MP Bayer8 單張約 6.3 MB，上限給 64 MB
net.core.rmem_max = 67108864
net.core.netdev_max_backlog = 10000
EOF
    sudo sysctl -p /etc/sysctl.d/60-aravis-gige.conf
fi

# ─────────── 5. arv-camera-test 權限 ───────────
log "arv-camera-test-${API} 加上 cap_net_raw"
CAMTEST="${PREFIX}/bin/arv-camera-test-${API}"
sudo setcap cap_net_raw+ep "$CAMTEST"
getcap "$CAMTEST"

# ─────────── 6. Python venv（帶 cap_net_raw） ───────────
if [[ $MAKE_VENV -eq 1 ]]; then
    log "建立 ${VENV_DIR}（--copies，共用系統的 gi / numpy / cv2）"
    python3 -m venv --copies --system-site-packages "$VENV_DIR"
    VENV_PY="$(readlink -f "${VENV_DIR}/bin/python3")"
    [[ "$VENV_PY" == "$VENV_DIR"/* ]] || die "venv 直譯器不是複製檔（${VENV_PY}），不對系統 python 加權限"
    # 只對這份複製的直譯器開放 raw socket，不影響系統 python
    sudo setcap cap_net_raw+ep "$VENV_PY"
    getcap "$VENV_PY"
    PY="${VENV_DIR}/bin/python3"
else
    PY="python3"
fi

# ─────────── 7. 驗證 ───────────
log "驗證安裝"
"${PREFIX}/bin/arv-tool-${API}" --version
"$PY" - <<EOF
import gi
gi.require_version('Aravis', '${API}')
from gi.repository import Aravis
import cv2, numpy
print(f"Aravis {Aravis.get_major_version()}.{Aravis.get_minor_version()}.{Aravis.get_micro_version()}"
      f" | OpenCV {cv2.__version__} | NumPy {numpy.__version__}")
EOF

log "完成"
cat <<EOF
下一步：
  1. 設定相機網口：   bash setup_gige_network.sh --ip 192.168.10.1/24
  2. 執行相機測試：   bash test_camera.sh
新開的 shell 會自動載入 /etc/profile.d/aravis.sh；目前的 shell 請先執行：
  source /etc/profile.d/aravis.sh
EOF
