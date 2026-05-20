#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# llama_pi.sh — llama.cpp RPC cluster on Raspberry Pi
#
# Topology: master + N workers (star topology)
#   Master: rpc-server + llama-server + nginx (API on port 52415)
#   Worker: rpc-server only (no model, no API)
#
# Automatic discovery via mDNS (_llama-rpc._tcp).
# Workers announce themselves; master detects and manages the cluster.
#
# Model loaded once on master, layers distributed to workers via RPC.
# RAM usage: model_size / (workers + 1) per node.
#
# Usage:
#   ./llama_pi.sh install master
#   ./llama_pi.sh install worker
#   ./llama_pi.sh load <hf-gguf-url>
#   ./llama_pi.sh add-worker <ip>      (manual override)
#   ./llama_pi.sh remove-worker <ip>   (manual override)
#   ./llama_pi.sh list-workers
#   ./llama_pi.sh status
#   ./llama_pi.sh uninstall
# ============================================================

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
CLUSTER_CONF="/etc/llama-cluster.conf"

RPC_PORT="${RPC_PORT:-50052}"
API_PORT="${API_PORT:-11434}"
PUBLIC_PORT="${PUBLIC_PORT:-52415}"
THREADS="${THREADS:-4}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${CYAN}══ $* ══${NC}"; }

# ── Shared build steps ────────────────────────────────────────
install_system_deps() {
  sudo apt-get update -qq
  sudo apt-get install -y -qq git cmake ninja-build libopenblas-dev \
    python3 python3-pip python3-venv g++ gcc curl
}

build_rpc_server() {
  if [ -d "$LLAMA_DIR" ]; then
    cd "$LLAMA_DIR" && git pull
  else
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  fi
  cd "$LLAMA_DIR"
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_RPC=ON -DGGML_NATIVE=ON -G Ninja
  cmake --build build --target rpc-server -j4
}

build_llama_server() {
  cd "$LLAMA_DIR"
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_RPC=ON -DGGML_NATIVE=ON -G Ninja
  cmake --build build --target llama-server -j4
}

install_python_venv() {
  python3 -m venv "$HOME/.venv-discovery"
  "$HOME/.venv-discovery/bin/pip" install -q zeroconf
}

# Create the mDNS discovery script
create_discovery_script() {
  cat > "$HOME/llama-discovery.py" << 'PYEOF'
#!/usr/bin/env python3
"""
llama-discovery.py — mDNS peer discovery for llama.cpp RPC cluster

Modes:
  announce  (worker):  announces _llama-rpc._tcp via mDNS
  listen    (master):  listens for announcements, manages /etc/llama-cluster.conf

Usage:
  python3 llama-discovery.py announce  --port 50052
  python3 llama-discovery.py listen    --port 50052
"""

import os, sys, time, signal, socket, subprocess, logging, json
from pathlib import Path
from typing import Optional
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf, IPVersion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("discovery")

SERVICE_TYPE = "_llama-rpc._tcp.local."
CLUSTER_CONF = Path("/etc/llama-cluster.conf")
RPC_PORT = 50052
llama_proc: Optional[subprocess.Popen] = None
peers: dict[str, str] = {}

def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

# ── Write /etc/llama-cluster.conf from peer dict ─────────────
def write_cluster_conf():
    addrs = list(peers.values())
    CLUSTER_CONF.write_text("\n".join(addrs) + "\n" if addrs else "")

# ── Restart llama-server (master only) ───────────────────────
def restart_llama_server():
    global llama_proc
    if llama_proc and llama_proc.poll() is None:
        log.info("Restarting llama-server with new worker list...")
        llama_proc.terminate()
        try:
            llama_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llama_proc.kill()
    # Actual llama-server is managed by systemd; just signal it
    subprocess.run(["sudo", "systemctl", "restart", "llama-server"],
        capture_output=True)

# ── Listener (master mode) ───────────────────────────────────
class MasterListener:
    def __init__(self, my_ip: str):
        self.my_ip = my_ip

    def add_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if not info:
            return
        node_id = name.split(".")[0]
        ip = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
        if not ip or ip == self.my_ip:
            return
        addr = f"{ip}:{info.port}"
        if addr != peers.get(node_id):
            peers[node_id] = addr
            write_cluster_conf()
            log.info(f"Worker UP:   {node_id} @ {addr}  (cluster: {len(peers)})")
            restart_llama_server()

    def remove_service(self, zc, type_, name):
        node_id = name.split(".")[0]
        if node_id in peers:
            addr = peers.pop(node_id)
            write_cluster_conf()
            log.info(f"Worker DOWN: {node_id} @ {addr}  (cluster: {len(peers)})")
            restart_llama_server()

    def update_service(self, zc, type_, name):
        self.add_service(zc, type_, name)

# ── Main ──────────────────────────────────────────────────────
def main():
    global RPC_PORT
    mode = "announce" if "announce" in sys.argv else "listen"
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            RPC_PORT = int(sys.argv[i + 1])

    my_ip = get_local_ip()
    node_id = socket.gethostname().replace(".", "-")
    log.info(f"Mode: {mode}  Node: {node_id}  IP: {my_ip}  RPC: {RPC_PORT}")

    zc = Zeroconf(ip_version=IPVersion.V4Only)

    if mode == "announce":
        service_info = ServiceInfo(
            SERVICE_TYPE,
            f"{node_id}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(my_ip)],
            port=RPC_PORT,
            properties={"hostname": socket.gethostname()},
        )
        zc.register_service(service_info)
        log.info(f"Announced as {node_id}.{SERVICE_TYPE}")
        signal.pause()
    else:
        log.info(f"Listening for _llama-rpc._tcp workers on {my_ip}...")
        ServiceBrowser(zc, SERVICE_TYPE, MasterListener(my_ip))
        signal.pause()

if __name__ == "__main__":
    main()
PYEOF
  chmod +x "$HOME/llama-discovery.py"
}

# ── install master ────────────────────────────────────────────
install_master() {
  section "Master install — Step 1/6: System dependencies"
  install_system_deps
  sudo apt-get install -y -qq nginx

  section "Master install — Step 2/6: Build llama.cpp"
  build_rpc_server
  build_llama_server

  section "Master install — Step 3/6: Python discovery venv"
  install_python_venv
  create_discovery_script

  section "Master install — Step 4/6: systemd services"

  sudo tee /etc/systemd/system/llama-rpc.service > /dev/null << SERVICEEOF
[Unit]
Description=llama.cpp RPC worker (master)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
ExecStart=$LLAMA_DIR/build/bin/rpc-server --host 0.0.0.0 --port $RPC_PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

  # ── llama-server wrapper ──────────────────────────────────────
  sudo tee /usr/local/bin/llama-server-wrapper.sh > /dev/null << 'WRAPEOF'
#!/usr/bin/env bash
set -euo pipefail
CONF="/etc/llama-server.env"
CLUSTER="/etc/llama-cluster.conf"

if [ ! -f "$CONF" ]; then
  echo "[llama-wrapper] No /etc/llama-server.env — run 'llama_pi.sh load <url>'"
  exit 1
fi
source "$CONF"

if [ ! -f "$MODEL_PATH" ]; then
  echo "[llama-wrapper] Model not found: $MODEL_PATH"
  exit 1
fi

WORKER_ARGS=()
if [ -f "$CLUSTER" ]; then
  while IFS= read -r entry; do
    [ -z "$entry" ] && continue
    # append default RPC port if missing
    if [[ "$entry" != *:* ]]; then
      entry="${entry}:${RPC_PORT}"
    fi
    WORKER_ARGS+=("--rpc" "$entry")
  done < "$CLUSTER"
fi

exec "$LLAMA_DIR/build/bin/llama-server" \
  --model "$MODEL_PATH" \
  --alias "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port "$API_PORT" \
  --threads "$THREADS" \
  --ctx-size 2048 \
  --fit off \
  "${WORKER_ARGS[@]}"
WRAPEOF
  sudo chmod +x /usr/local/bin/llama-server-wrapper.sh

  sudo tee /etc/systemd/system/llama-server.service > /dev/null << SERVICEEOF
[Unit]
Description=llama.cpp inference server (master)
After=network-online.target llama-rpc.service
Wants=network-online.target
Requires=llama-rpc.service

[Service]
Type=simple
User=$USER
Environment=LLAMA_DIR=$LLAMA_DIR
Environment=RPC_PORT=$RPC_PORT
Environment=API_PORT=$API_PORT
Environment=THREADS=$THREADS
ExecStart=/usr/local/bin/llama-server-wrapper.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICEEOF

  # ── mDNS discovery daemon (listener) ──────────────────────────
  sudo tee /etc/systemd/system/llama-discovery.service > /dev/null << SERVICEEOF
[Unit]
Description=llama.cpp mDNS discovery (master)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
ExecStart=$HOME/.venv-discovery/bin/python3 $HOME/llama-discovery.py listen --port $RPC_PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

  section "Master install — Step 5/6: nginx reverse proxy"
  sudo tee /etc/nginx/sites-available/llama > /dev/null << NGINXEOF
server {
    listen $PUBLIC_PORT;
    server_name _;
    proxy_read_timeout 300s;
    proxy_connect_timeout 10s;
    proxy_send_timeout 300s;
    proxy_buffering off;
    proxy_set_header Host \$host;
    proxy_set_header X-Real-IP \$remote_addr;
    proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    location /v1/ { proxy_pass http://127.0.0.1:$API_PORT; }
    location /api/ { proxy_pass http://127.0.0.1:$API_PORT; }
    location /health { proxy_pass http://127.0.0.1:$API_PORT/health; }
    location / { proxy_pass http://127.0.0.1:$API_PORT; }
}
NGINXEOF
  sudo ln -sf /etc/nginx/sites-available/llama /etc/nginx/sites-enabled/llama
  sudo rm -f /etc/nginx/sites-enabled/default
  sudo nginx -t && sudo systemctl reload nginx

  sudo systemctl daemon-reload
  sudo systemctl enable llama-rpc.service llama-server.service llama-discovery.service

  # Init config files
  sudo touch "$CLUSTER_CONF"
  sudo tee /etc/llama-server.env > /dev/null << ENVEOF
MODEL_PATH=""
MODEL_NAME=""
LLAMA_DIR=$LLAMA_DIR
RPC_PORT=$RPC_PORT
API_PORT=$API_PORT
THREADS=$THREADS
ENVEOF

  section "Master install — Step 6/6: Helper scripts"
  cat > "$HOME/llama-status.sh" << 'STATUSEOF'
#!/usr/bin/env bash
RPC_PORT="${RPC_PORT:-50052}"
API_PORT="${API_PORT:-11434}"
PUBLIC_PORT="${PUBLIC_PORT:-52415}"

# ── detect master IP ──────────────────────────────────────────
MASTER_IP=$(ip -4 route get 8.8.8.8 2>/dev/null | head -1 | awk '{print $7}')
[ -z "$MASTER_IP" ] && MASTER_IP="<MASTER_IP>"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║             llama.cpp RPC Cluster Status                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Services ────────────────────────────────────────────────────
echo "  Services"
echo "  ────────────────────────────────────────────────────────"
for svc in llama-rpc llama-server llama-discovery nginx; do
  state=$(systemctl is-active "$svc" 2>/dev/null)
  icon=$([ "$state" = "active" ] && echo "  ✓" || echo "  ✗")
  printf "    %-22s %s  %s\n" "$svc" "$icon" "${state:-inactive}"
done
echo ""

# ── Model ───────────────────────────────────────────────────────
if [ -f /etc/llama-server.env ]; then
  source /etc/llama-server.env
  echo "  Model:        ${MODEL_NAME:-not loaded}"
fi
echo ""

# ── Workers ─────────────────────────────────────────────────────
nw=0
[ -f /etc/llama-cluster.conf ] && nw=$(wc -l < /etc/llama-cluster.conf | tr -d ' ')
echo "  🔗 Cluster ($nw worker$( [ "$nw" -ne 1 ] && echo 's'))"
echo "  ────────────────────────────────────────────────────────"
if [ "$nw" -gt 0 ]; then
  while IFS= read -r ip; do
    [ -z "$ip" ] && continue
    echo "    ⚡ $ip"
  done < /etc/llama-cluster.conf
  echo ""
  total_nodes=$((nw + 1))
  echo "    Model split across $total_nodes nodes"
else
  echo "    (no workers yet — waiting for mDNS announcements)"
  echo ""
fi
echo ""

# ── Quick commands ──────────────────────────────────────────────
echo "   Commands"
echo "  ────────────────────────────────────────────────────────"
echo ""
echo "    Start cluster:"
echo "      sudo systemctl start llama-rpc llama-discovery"
echo ""
echo "    Load a model:"
echo "      ./llama_pi.sh load <hf-gguf-url>"
echo ""
echo "    Start inference (after model loaded):"
echo "      sudo systemctl start llama-server"
echo ""
echo "    Stop inference:"
echo "      sudo systemctl stop llama-server"
echo ""
echo "    Stop cluster:"
echo "      sudo systemctl stop llama-server llama-discovery llama-rpc"
echo ""
echo "    Logs:"
echo "      journalctl -u llama-rpc -f        # RPC worker status"
echo "      journalctl -u llama-discovery -f  # mDNS discovery events"
echo "      journalctl -u llama-server -f     # inference logs"
echo ""
echo "    Workers (/etc/llama-cluster.conf):"
echo "      ./llama_pi.sh list-workers"
echo "      ./llama_pi.sh add-worker <ip>"
echo "      ./llama_pi.sh remove-worker <ip>"
echo ""

# ── Test command ────────────────────────────────────────────────
echo "  🧪 Test inference"
echo "  ────────────────────────────────────────────────────────"
echo ""
cat <<CURLDEMO
    curl http://${MASTER_IP}:${PUBLIC_PORT}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "max_tokens": 200
      }'
CURLDEMO
echo ""
STATUSEOF
  chmod +x "$HOME/llama-status.sh"

  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║         Master install complete!                         ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
  echo "     Next steps:"
  echo ""
  echo "  1. Start the RPC worker + mDNS listener:"
  echo "       sudo systemctl start llama-rpc llama-discovery"
  echo ""
  echo "  2. Install workers on other Pis:"
  echo "       ./llama_pi.sh install worker"
  echo "       sudo systemctl start llama-rpc llama-discovery"
  echo ""
  echo "  3. Workers auto-register via mDNS."
  echo "     Verify with: ~/llama-status.sh"
  echo ""
  echo "  4. Load a model:"
  echo "       ./llama_pi.sh load <hf-gguf-url>"
  echo ""
  echo "  5. Start inference:"
  echo "       sudo systemctl start llama-server"
  echo ""
  echo "  6. Test from another machine:"
  echo "       curl http://<MASTER_IP>:$PUBLIC_PORT/v1/chat/completions \\"
  echo "         -H \"Content-Type: application/json\" \\"
  echo "         -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Tell me a joke\"}], \"max_tokens\": 200}'"
  echo ""
  echo "  ═══════════════════════════════════════════════════════"
}

# ── install worker ────────────────────────────────────────────
install_worker() {
  section "Worker install — Step 1/4: System dependencies"
  install_system_deps

  section "Worker install — Step 2/4: Build rpc-server"
  build_rpc_server

  section "Worker install — Step 3/4: mDNS announcer"
  install_python_venv
  create_discovery_script

  sudo tee /etc/systemd/system/llama-rpc.service > /dev/null << SERVICEEOF
[Unit]
Description=llama.cpp RPC worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
ExecStart=$LLAMA_DIR/build/bin/rpc-server --host 0.0.0.0 --port $RPC_PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

  sudo tee /etc/systemd/system/llama-discovery.service > /dev/null << SERVICEEOF
[Unit]
Description=llama.cpp mDNS announcer (worker)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
ExecStart=$HOME/.venv-discovery/bin/python3 $HOME/llama-discovery.py announce --port $RPC_PORT
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

  section "Worker install — Step 4/4: systemd enable + helper scripts"
  sudo systemctl daemon-reload
  sudo systemctl enable llama-rpc.service llama-discovery.service

  # ── worker status script ────────────────────────────────────
  cat > "$HOME/llama-status.sh" << 'WORKERSTATUS'
#!/usr/bin/env bash
RPC_PORT="${RPC_PORT:-50052}"
MY_IP=$(ip -4 route get 8.8.8.8 2>/dev/null | head -1 | awk '{print $7}')
[ -z "$MY_IP" ] && MY_IP="<IP>"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║       llama.cpp RPC Worker Status                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

echo "  Node:        $(hostname)"
echo "  IP:          $MY_IP"
echo "  RPC Port:    $RPC_PORT"
echo ""

echo "  Services"
echo "  ────────────────────────────────────────────────────────"
for svc in llama-rpc llama-discovery; do
  state=$(systemctl is-active "$svc" 2>/dev/null)
  icon=$([ "$state" = "active" ] && echo "  ✓" || echo "  ✗")
  printf "    %-22s %s  %s\n" "$svc" "$icon" "${state:-inactive}"
done
echo ""

echo "   Commands"
echo "  ────────────────────────────────────────────────────────"
echo ""
echo "    Start:"
echo "      sudo systemctl start llama-rpc llama-discovery"
echo ""
echo "    Stop:"
echo "      sudo systemctl stop llama-rpc llama-discovery"
echo ""
echo "    Logs:"
echo "      journalctl -u llama-rpc -f        # RPC worker status"
echo "      journalctl -u llama-discovery -f  # mDNS announcement events"
echo ""
echo "    Re-announce to master (restart mDNS):"
echo "      sudo systemctl restart llama-discovery"
echo ""
WORKERSTATUS
  chmod +x "$HOME/llama-status.sh"

  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║         Worker install complete!                         ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo ""
  echo "     Next steps:"
  echo ""
  echo "  1. Start the RPC worker + mDNS announcer:"
  echo "       sudo systemctl start llama-rpc llama-discovery"
  echo ""
  echo "  2. The master will auto-discover this node via mDNS."
  echo "     Verify with: ~/llama-status.sh"
  echo ""
  echo "  Logs:"
  echo "    journalctl -u llama-rpc -f"
  echo "    journalctl -u llama-discovery -f"
  echo "  ═══════════════════════════════════════════════════════"
}

# ── load model ────────────────────────────────────────────────
load_model() {
  local url="${1:-}"
  if [ -z "$url" ]; then
    error "Usage: $0 load <hf-gguf-url>"
  fi
  MODEL_FILE=$(basename "$url" | cut -d'?' -f1)
  MODEL_NAME=$(echo "$MODEL_FILE" | sed 's/\.gguf$//' | sed 's/-Q[0-9].*//' | tr '[:upper:]' '[:lower:]')
  MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
  mkdir -p "$MODEL_DIR"

  if [ -f "$MODEL_PATH" ]; then
    magic=$(head -c 4 "$MODEL_PATH" 2>/dev/null || echo "")
    if [ "$magic" != "GGUF" ]; then
      warn "Corrupt file, re-downloading..."
      rm -f "$MODEL_PATH"
    fi
  fi
  if [ -f "$MODEL_PATH" ]; then
    info "Model already cached: $MODEL_FILE"
  else
    info "Downloading $MODEL_FILE..."
    curl -L --progress-bar -C - -o "$MODEL_PATH" "$url"
  fi
  magic=$(head -c 4 "$MODEL_PATH" 2>/dev/null || echo "")
  if [ "$magic" != "GGUF" ]; then
    error "Not a valid GGUF file (starts with '$magic')"
  fi

  sudo tee /etc/llama-server.env > /dev/null << ENVEOF
MODEL_PATH="$MODEL_PATH"
MODEL_NAME="$MODEL_NAME"
LLAMA_DIR=$LLAMA_DIR
RPC_PORT=$RPC_PORT
API_PORT=$API_PORT
THREADS=$THREADS
ENVEOF
  info "Model configured: $MODEL_NAME"

  if systemctl is-active llama-server >/dev/null 2>&1; then
    sudo systemctl restart llama-server
  else
    info "Start: sudo systemctl start llama-server"
  fi
}

# ── add/remove worker (manual) ────────────────────────────────
add_worker() {
  local entry="${1:-}"
  [ -z "$entry" ] && error "Usage: $0 add-worker <ip>[:<port>]"
  sudo touch "$CLUSTER_CONF"
  if [[ "$entry" != *:* ]]; then
    entry="${entry}:${RPC_PORT}"
  fi
  if grep -qxF "$entry" "$CLUSTER_CONF" 2>/dev/null; then
    info "Worker $entry already registered."; return
  fi
  echo "$entry" | sudo tee -a "$CLUSTER_CONF" > /dev/null
  info "Worker $entry added."
  systemctl is-active llama-server >/dev/null 2>&1 && sudo systemctl restart llama-server
}

remove_worker() {
  local ip="${1:-}"
  [ -z "$ip" ] && error "Usage: $0 remove-worker <ip>"
  sudo sed -i "\|^${ip}|d" "$CLUSTER_CONF" 2>/dev/null || true
  info "Worker $ip removed."
  systemctl is-active llama-server >/dev/null 2>&1 && sudo systemctl restart llama-server
}

list_workers() {
  if [ ! -f "$CLUSTER_CONF" ]; then
    echo "No workers registered."
    return
  fi
  nw=$(grep -c . "$CLUSTER_CONF" 2>/dev/null || echo 0)
  echo "Workers: $nw"
  while IFS= read -r ip; do
    [ -z "$ip" ] && continue
    echo "  - $ip"
  done < "$CLUSTER_CONF"
}

# ── uninstall ─────────────────────────────────────────────────
uninstall() {
  info "Stopping services..."
  for svc in llama-discovery llama-server llama-rpc; do
    sudo systemctl stop "$svc" 2>/dev/null || true
    sudo systemctl disable "$svc" 2>/dev/null || true
  done
  info "Removing systemd files..."
  sudo rm -f /etc/systemd/system/llama-rpc.service
  sudo rm -f /etc/systemd/system/llama-server.service
  sudo rm -f /etc/systemd/system/llama-discovery.service
  sudo rm -f /usr/local/bin/llama-server-wrapper.sh
  sudo systemctl daemon-reload
  info "Removing nginx config..."
  sudo rm -f /etc/nginx/sites-enabled/llama
  sudo rm -f /etc/nginx/sites-available/llama
  sudo systemctl reload nginx 2>/dev/null || true
  info "Removing build, models, config..."
  rm -rf "$LLAMA_DIR" "$MODEL_DIR"
  sudo rm -f "$CLUSTER_CONF" /etc/llama-server.env
  rm -f "$HOME/llama-discovery.py" "$HOME/llama-status.sh"
  rm -rf "$HOME/.venv-discovery"
  info "Uninstall complete."
}

# ── status ────────────────────────────────────────────────────
status() {
  if [ -f "$HOME/llama-status.sh" ]; then
    bash "$HOME/llama-status.sh"
  else
    error "Not installed."
  fi
}

# ── main ──────────────────────────────────────────────────────
cmd="${1:-}"
subcmd="${2:-}"

case "$cmd" in
  install)
    case "$subcmd" in
      master) install_master ;;
      worker) install_worker ;;
      *) echo "Usage: $0 install {master | worker}" && exit 1 ;;
    esac
    ;;
  load) load_model "${2:-}" ;;
  add-worker) add_worker "${2:-}" ;;
  remove-worker) remove_worker "${2:-}" ;;
  list-workers) list_workers ;;
  status) status ;;
  uninstall)
    warn "This will remove llama.cpp, all models and config."
    read -rp "Continue? [y/N] " confirm
    case "$confirm" in
      y|Y|yes|Yes) uninstall ;;
      *) echo "Aborted." ;;
    esac
    ;;
  *)
    echo "Usage: $0 {install master|worker | load <url> | add-worker <ip> | remove-worker <ip> | list-workers | status | uninstall}"
    exit 1
    ;;
esac
