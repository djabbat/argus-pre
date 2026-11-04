#!/usr/bin/env sh
# ARGUS-OS1 installer — build from source and install to ~/.local/bin
# Usage: curl -fsSL https://<host>/install.sh | sh
#        ARGUS_INSTALL_DIR=~/.local/bin sh install.sh
set -eu

INSTALL_DIR="${ARGUS_INSTALL_DIR:-$HOME/.local/bin}"
SRC_DIR="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"

echo "═══ ARGUS-OS1 installer ═══"

if [ -f "target/release/argus-agent" ]; then
  BIN="target/release/argus-agent"
elif [ -f "$SRC_DIR/target/release/argus-agent" ]; then
  BIN="$SRC_DIR/target/release/argus-agent"
else
  echo "→ building from source (cargo build --release)..."
  if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ Rust toolchain not found. Install via https://rustup.rs" >&2
    exit 1
  fi
  cargo build --release --manifest-path "$SRC_DIR/Cargo.toml" --bin argus-agent
  BIN="$SRC_DIR/target/release/argus-agent"
fi

mkdir -p "$INSTALL_DIR"
cp "$BIN" "$INSTALL_DIR/argus-agent"
ln -sf argus-agent "$INSTALL_DIR/atag"

echo "✅ installed: $INSTALL_DIR/argus-agent (alias: atag)"
echo "   state dir: ~/.argus-agent (override ARGUS_STATE_DIR)"
echo "   run:       argus-agent init   # memory + starter skills"
