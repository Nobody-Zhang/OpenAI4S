#!/usr/bin/env bash
# OpenAI4S · build a self-contained macOS .app + .dmg for release.
#
# Strategy: this project's kernel spawns its worker via
#   subprocess.Popen([sys.executable, "-u", worker.py], PYTHONPATH=<repo root>)
# so freezing (py2app / PyInstaller) would break it — sys.executable must stay a
# real interpreter and the package must stay loose .py files on disk. We therefore
# embed a *relocatable* standalone CPython (python-build-standalone, via uv) plus
# the full CORE_PACKAGES science stack, and ship the source tree intact. Every
# Path(__file__)-relative lookup (webui/, skills/, envs/, compute/templates/,
# worker.py) then resolves correctly wherever the .app lives, and all writable
# state goes to ~/.openai4s (outside the read-only bundle).
#
# No Apple Developer credentials are used: the app is ad-hoc signed only (free),
# which is still required so Apple Silicon does not kill an unsigned binary.
set -euo pipefail

APP_NAME="OpenAI4S"
VERSION="0.1.0"
ARCH="arm64"          # user-facing macOS arch label (dmg name / notes)
PYARCH="aarch64"      # python-build-standalone / uv arch token
PYSERIES="3.13"
BUNDLE_ID="com.openai4s.app"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BUILD="${BUILD:-$REPO_ROOT/.build/dmg}"
DIST="${DIST:-$REPO_ROOT/dist}"

STAGE="$BUILD/stage"
APP="$STAGE/$APP_NAME.app"
CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
RUNTIME="$RES/runtime"
SRC="$RES/src"
DMG="$DIST/$APP_NAME-$VERSION-macos-$ARCH.dmg"

echo "== OpenAI4S macOS packaging =="
echo "  repo    : $REPO_ROOT"
echo "  build   : $BUILD"
echo "  dmg out : $DMG"

# --------------------------------------------------------------------------- #
# 0) locate a relocatable standalone CPython (python-build-standalone via uv)
# --------------------------------------------------------------------------- #
echo "-- [0/9] locating standalone CPython $PYSERIES ($ARCH) --"
uv python install "$PYSERIES" >/dev/null 2>&1 || true
STDPY_BIN="$(ls -d "$HOME/.local/share/uv/python/"cpython-"$PYSERIES"*-macos-"$PYARCH"-none/bin/python3 2>/dev/null | sort | tail -1 || true)"
if [ -z "${STDPY_BIN:-}" ] || [ ! -x "$STDPY_BIN" ]; then
  echo "error: could not find a uv-managed standalone CPython $PYSERIES for $PYARCH" >&2
  echo "       looked under: $HOME/.local/share/uv/python/cpython-$PYSERIES*-macos-$PYARCH-none/bin/python3" >&2
  exit 1
fi
STDPY_ROOT="$(cd "$(dirname "$STDPY_BIN")/.." && pwd)"
echo "   using: $STDPY_ROOT"

# --------------------------------------------------------------------------- #
# 1) clean & skeleton
# --------------------------------------------------------------------------- #
echo "-- [1/9] cleaning & creating bundle skeleton --"
rm -rf "$BUILD"
mkdir -p "$RUNTIME" "$SRC" "$CONTENTS/MacOS" "$DIST"

# --------------------------------------------------------------------------- #
# 2) copy the runtime (preserving symlinks) and prune non-runtime bulk
# --------------------------------------------------------------------------- #
echo "-- [2/9] copying embedded Python runtime --"
cp -R "$STDPY_ROOT/." "$RUNTIME/"
rm -f "$RUNTIME/BUILD" 2>/dev/null || true
rm -rf "$RUNTIME"/lib/python*/test "$RUNTIME"/lib/python*/idlelib \
       "$RUNTIME"/lib/python*/turtledemo "$RUNTIME"/lib/python*/lib2to3 2>/dev/null || true
find "$RUNTIME" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
RUNPY="$RUNTIME/bin/python3"
echo "   runtime python: $("$RUNPY" -c 'import sys;print(sys.version.split()[0])')"

# --------------------------------------------------------------------------- #
# 3) pre-bake the full CORE_PACKAGES science stack (matches
#    openai4s/kernel/preinstall.py::CORE_PACKAGES) so ensure_core() at startup
#    is a no-op — no network, no writing into the read-only bundle.
# --------------------------------------------------------------------------- #
echo "-- [3/9] installing the CORE science stack into the runtime (this is the slow step) --"
# python-build-standalone ships a PEP 668 marker; drop it on our private copy so
# pip may install into the bundled interpreter's own site-packages.
rm -f "$RUNTIME"/lib/python*/EXTERNALLY-MANAGED 2>/dev/null || true
"$RUNPY" -m pip install --upgrade --no-warn-script-location pip >/dev/null
"$RUNPY" -m pip install --no-warn-script-location \
  numpy pandas scipy matplotlib seaborn scikit-learn statsmodels sympy networkx \
  biopython pillow requests httpx beautifulsoup4 lxml openpyxl tabulate tqdm pyyaml \
  plotly h5py pyarrow python-dateutil regex
# drop pip's own wheel cache dirs that may have landed inside the runtime
find "$RUNTIME" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# --------------------------------------------------------------------------- #
# 4) copy the source tree intact (loose .py) — every relative lookup depends
#    on openai4s/ + openai4s_compute_provider/ + skills/ + envs/ being siblings
# --------------------------------------------------------------------------- #
echo "-- [4/9] copying source tree --"
rsync -a \
  --exclude '.git' --exclude '.venv' --exclude '.build' --exclude 'dist' \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  --exclude '*.egg-info' --exclude '.env' --exclude '.env.*' --exclude '!.env.example' \
  --exclude '*.db' --exclude '.DS_Store' --exclude 'tests' --exclude 'readme-gifs-hd' \
  --exclude '.claude' \
  "$REPO_ROOT/openai4s" "$REPO_ROOT/openai4s_compute_provider" \
  "$REPO_ROOT/envs" "$REPO_ROOT/skills" "$REPO_ROOT/scripts" "$REPO_ROOT/docs" \
  "$SRC/"
cp "$REPO_ROOT/README.md" "$REPO_ROOT/README_zh.md" "$REPO_ROOT/LICENSE" \
   "$REPO_ROOT/.env.example" "$REPO_ROOT/pyproject.toml" "$SRC/" 2>/dev/null || true

# --------------------------------------------------------------------------- #
# 5) launcher (CFBundleExecutable). exec replaces the shell so the process macOS
#    tracks as the app *is* the python server — Quit delivers SIGTERM to it, and
#    cmd_serve's SIGTERM handler shuts down cleanly.
# --------------------------------------------------------------------------- #
echo "-- [5/9] writing launcher --"
cat > "$CONTENTS/MacOS/$APP_NAME" <<'LAUNCHER'
#!/bin/bash
# OpenAI4S launcher — starts the local daemon + web UI and opens the browser.
set -e
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
PY="$RES/runtime/bin/python3"
SRC="$RES/src"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI4S_DATA_DIR="${OPENAI4S_DATA_DIR:-$HOME/.openai4s}"
export MPLBACKEND="Agg"
# stdlib-only core; keep PATH sane for webbrowser's `open` when launched from Finder
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

mkdir -p "$OPENAI4S_DATA_DIR/logs"
cd "$OPENAI4S_DATA_DIR"
exec "$PY" -m openai4s serve >>"$OPENAI4S_DATA_DIR/logs/app.out" 2>&1
LAUNCHER
chmod +x "$CONTENTS/MacOS/$APP_NAME"

# --------------------------------------------------------------------------- #
# 6) Info.plist + PkgInfo
# --------------------------------------------------------------------------- #
echo "-- [6/9] writing Info.plist --"
cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>app.icns</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
</dict>
</plist>
PLIST
printf 'APPL????' > "$CONTENTS/PkgInfo"

# --------------------------------------------------------------------------- #
# 7) app icon (best-effort; never fails the build)
# --------------------------------------------------------------------------- #
echo "-- [7/9] generating app icon (best-effort) --"
(
  set -e
  "$RUNPY" - "$BUILD/icon_1024.png" <<'PYICON'
import sys
from PIL import Image, ImageDraw, ImageFont
size = 1024
img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
# rounded square, deep-space gradient-ish two tone
for y in range(size):
    t = y / size
    r = int(18 + 24 * t); g = int(20 + 30 * t); b = int(40 + 70 * t)
    d.line([(0, y), (size, y)], fill=(r, g, b, 255))
mask = Image.new("L", (size, size), 0)
ImageDraw.Draw(mask).rounded_rectangle([0, 0, size, size], radius=200, fill=255)
img.putalpha(mask)
d = ImageDraw.Draw(img)
def font(sz):
    for p in ("/System/Library/Fonts/SFNSRounded.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()
txt = "O4S"
f = font(400)
bb = d.textbbox((0, 0), txt, font=f)
w, h = bb[2] - bb[0], bb[3] - bb[1]
d.text(((size - w) / 2 - bb[0], (size - h) / 2 - bb[1] - 40), txt,
       font=f, fill=(240, 245, 255, 255))
fs = font(90)
sub = "OpenAI4S"
bb2 = d.textbbox((0, 0), sub, font=fs)
d.text(((size - (bb2[2]-bb2[0])) / 2 - bb2[0], size - 260), sub,
       font=fs, fill=(150, 170, 210, 255))
img.save(sys.argv[1])
PYICON
  ICONSET="$BUILD/$APP_NAME.iconset"
  mkdir -p "$ICONSET"
  for s in 16 32 128 256 512; do
    sips -z $s $s "$BUILD/icon_1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z $d $d "$BUILD/icon_1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$RES/app.icns"
  echo "   icon written: $RES/app.icns"
) || echo "   icon generation skipped (non-fatal)"

# --------------------------------------------------------------------------- #
# 8) ad-hoc codesign (no Apple Developer credentials; required on Apple Silicon)
# --------------------------------------------------------------------------- #
echo "-- [8/9] ad-hoc codesigning --"
codesign --force --deep --sign - --timestamp=none "$APP" 2>&1 | tail -2 || true
codesign --verify --deep "$APP" && echo "   codesign verify: OK" || echo "   codesign verify: WARN (ad-hoc)"

# --------------------------------------------------------------------------- #
# 9) build the DMG
# --------------------------------------------------------------------------- #
echo "-- [9/9] building DMG --"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/READ ME — first launch.txt" <<'NOTE'
OpenAI4S — first launch on macOS
================================

1. Drag OpenAI4S.app onto the Applications folder (shown here).

2. This build is ad-hoc signed but NOT notarized (no Apple Developer account),
   so Gatekeeper will warn on first launch. To open it:
     • Right-click (or Control-click) OpenAI4S.app → Open → Open, OR
     • run once in Terminal:  xattr -dr com.apple.quarantine /Applications/OpenAI4S.app

3. Launching starts a local server and opens http://127.0.0.1:8760/ in your
   browser. Set your LLM provider + API key in the UI (Customize → Models).
   All data lives in ~/.openai4s.  Logs: ~/.openai4s/logs/app.out

4. Apple Silicon only. The scientific stack (numpy/pandas/scipy/matplotlib/…)
   is bundled and works offline.
NOTE

rm -f "$DMG"
hdiutil create -volname "$APP_NAME $VERSION" -srcfolder "$STAGE" \
  -ov -format UDZO "$DMG" >/dev/null
echo
echo "== DONE =="
echo "  app  : $(du -sh "$APP" | cut -f1)   $APP"
echo "  dmg  : $(du -h "$DMG" | cut -f1)   $DMG"
