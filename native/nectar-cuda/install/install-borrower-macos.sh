#!/bin/sh
# Installs Nectar on a Mac that borrows GPUs, or removes it (--uninstall). Run with sudo, in the
# folder with Nectar's macOS build (or give it with --source). Not yet run on a Mac.
#
# - /usr/local/nectar/bin: nectar (the command), nectar-svc (the service), nectar-detect
# - /usr/local/nectar/lib/libnectar_vk.dylib: Nectar's Vulkan driver, registered with the Vulkan
#   loader in /usr/local/share/vulkan/icd.d. It lists no device while nothing is borrowed. Vulkan
#   programs need the loader, from LunarG's Vulkan SDK. There are no CUDA programs on macOS.
# - The co.nectar.borrow launchd service, which finds lenders, holds the lease and writes
#   borrow.cfg while a GPU is borrowed.
# - "/Library/Application Support/Nectar": readable by everyone, writable by root only. The shared
#   secret in it is readable by root only.
#
#   sudo ./install-borrower-macos.sh --secret "the secret entered on the lender"
#   sudo ./install-borrower-macos.sh --uninstall

set -eu

secret=""
uninstall=0
source=$(dirname "$0")
while [ $# -gt 0 ]; do
    case "$1" in
        --secret) secret="$2"; shift 2 ;;
        --uninstall) uninstall=1; shift ;;
        --source) source="$2"; shift 2 ;;
        *) echo "usage: $0 --secret <secret> [--source <folder>] | --uninstall" >&2; exit 2 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "Run it with sudo" >&2
    exit 1
fi

programs=/usr/local/nectar
state="/Library/Application Support/Nectar"
service=/Library/LaunchDaemons/co.nectar.borrow.plist
vulkan_manifest=/usr/local/share/vulkan/icd.d/nectar_icd.json

remove() {
    if [ -f "$service" ]; then
        # Stopping returns a borrowed GPU and removes borrow.cfg
        launchctl bootout system "$service" 2>/dev/null || true
        rm -f "$service"
    fi
    rm -f "$vulkan_manifest" /usr/local/bin/nectar
    rm -rf "$programs" "$state"
}

if [ "$uninstall" -eq 1 ]; then
    remove
    echo "Nectar removed"
    exit 0
fi

if [ ${#secret} -lt 8 ]; then
    echo "Give the lender's shared secret, at least 8 characters: --secret <secret>" >&2
    exit 2
fi
for file in nectar nectar-svc nectar-detect libnectar_vk.dylib; do
    if [ ! -f "$source/$file" ]; then
        echo "No $file in $source; give Nectar's macOS build with --source" >&2
        exit 1
    fi
done

remove
install -d -m 755 "$programs/bin" "$programs/lib" "$state" /usr/local/bin "$(dirname "$vulkan_manifest")"
for file in nectar nectar-svc nectar-detect; do
    install -m 755 "$source/$file" "$programs/bin/$file"
done
[ -f "$source/nectar-server" ] && install -m 755 "$source/nectar-server" "$programs/bin/"
install -m 755 "$source/libnectar_vk.dylib" "$programs/lib/libnectar_vk.dylib"
cat > "$vulkan_manifest" <<EOF
{"file_format_version":"1.0.1","ICD":{"library_path":"$programs/lib/libnectar_vk.dylib","api_version":"1.4.363"}}
EOF
ln -sf "$programs/bin/nectar" /usr/local/bin/nectar

umask 077
printf '%s' "$secret" > "$state/secret"
umask 022

# Who may borrow and return (nectar-svc opens its socket to this group)
dseditgroup -o read nectar >/dev/null 2>&1 || dseditgroup -o create nectar
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    dseditgroup -o edit -a "$SUDO_USER" -t user nectar
fi

cat > "$service" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>co.nectar.borrow</string>
    <key>ProgramArguments</key>
    <array><string>$programs/bin/nectar-svc</string><string>run</string></array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
</dict>
</plist>
EOF
chmod 644 "$service"
launchctl bootstrap system "$service"

echo "Nectar installed. Then: nectar status, nectar borrow <lender>"
