#!/usr/bin/env bash
# A desktop in the browser, for watching Isaac Sim with `-E headless=false`. `setup.sh --desktop`
# installs it as the arena-desktop systemd service, which runs this script in the foreground.
#
# Xvfb draws display :1 (1920x1080) with an xfce session; Isaac Sim still renders on the GPU and
# shows its window there. x11vnc shares the display, and noVNC serves it at http://127.0.0.1:6080.
# Both listen on this machine only, so the desktop has no password: reach it through an SSH tunnel
# such as `brev port-forward <instance> -p 6080:6080`.
set -u
export DISPLAY=:1
SOCKET=/tmp/.X11-unix/X1
PORT=6080

rm -f /tmp/.X1-lock "$SOCKET"
Xvfb :1 -screen 0 1920x1080x24 -nolisten tcp &
for _ in $(seq 50); do [[ -S $SOCKET ]] && break; sleep 0.2; done
[[ -S $SOCKET ]] || { echo "desktop.sh: Xvfb did not start" >&2; exit 1; }
xset s off -dpms # never blank the screen

# Without a panel config, xfce opens a first-start dialog; start from its default panel instead.
XFCONF=$HOME/.config/xfce4/xfconf/xfce-perchannel-xml
if [[ ! -f $XFCONF/xfce4-panel.xml && -f /etc/xdg/xfce4/panel/default.xml ]]; then
  mkdir -p "$XFCONF" && cp /etc/xdg/xfce4/panel/default.xml "$XFCONF/xfce4-panel.xml"
fi
# Compositing only adds work on a software display.
dbus-launch --exit-with-session sh -c \
  'xfconf-query -c xfwm4 -p /general/use_compositing -n -t bool -s false; exec startxfce4' &

x11vnc -display :1 -localhost -rfbport 5900 -nopw -forever -shared -quiet &

# noVNC, with an index page that opens the viewer connected and scaled to the browser window.
WEB=$HOME/.cache/arena-desktop-novnc
mkdir -p "$WEB"
for f in /usr/share/novnc/*; do ln -sfn "$f" "$WEB/"; done
rm -f "$WEB/index.html"
echo '<!doctype html><meta http-equiv="refresh" content="0; url=vnc.html?autoconnect=1&resize=scale">' >"$WEB/index.html"
websockify --web "$WEB" "127.0.0.1:$PORT" localhost:5900 &

# If any part exits, stop the rest and fail, so systemd restarts the whole desktop.
wait -n
echo "desktop.sh: a desktop process exited; restarting" >&2
kill $(jobs -p) 2>/dev/null
exit 1
