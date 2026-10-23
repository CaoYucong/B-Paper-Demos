#!/bin/bash
# Double-click in Finder to start the lab. Closing this window stops it.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
URL="http://localhost:8501/"

pause() {
  echo
  echo "按回车关闭这个窗口。"
  read -r _
}

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "找不到虚拟环境：$ROOT/.venv"
  echo "请先在终端执行："
  echo "  cd \"$ROOT\""
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/python -m pip install -r requirements.txt"
  pause
  exit 1
fi

if lsof -nP -iTCP:8501 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "实验室已经在运行。"
  echo "$URL"
  open "$URL"
  pause
  exit 0
fi

export PYTHONUNBUFFERED=1
"$ROOT/.venv/bin/python" "$ROOT/serve.py" &
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

echo "正在启动实验室…"
echo "浏览器会打开 $URL"
echo "关掉这个窗口就会停止服务。"
echo

ready=0
for _ in $(seq 1 80); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "启动失败。"
    wait "$server_pid" || true
    pause
    exit 1
  fi
  if curl -fsS -o /dev/null --max-time 1 "$URL" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.25
done

if [[ "$ready" -eq 1 ]]; then
  open "$URL"
else
  echo "服务还没有响应。可以稍后自己打开 $URL"
fi

wait "$server_pid"
status=$?
echo "服务已停止（退出码 $status）。"
pause
exit "$status"
