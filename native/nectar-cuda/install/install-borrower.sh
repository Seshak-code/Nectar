#!/bin/sh
# Installs Nectar on a Linux computer that borrows GPUs, or removes it (--uninstall). Run with
# sudo, in the folder with Nectar's Linux build (or give it with --source).
#
# - /opt/nectar/bin: nectar (the command), nectar-svc (the service), nectar-detect, cuda-probe
# - /opt/nectar/lib/libcuda.so.1: Nectar's CUDA driver, which every CUDA program loads, listed
#   first in /etc/ld.so.conf.d. While no GPU is borrowed it reports no device at once, like a
#   computer without one.
# - The nectar-borrow systemd service, which finds lenders, holds the lease and writes
#   /var/lib/nectar/borrow.cfg while a GPU is borrowed.
# - /opt/nectar/lib/libnectar_vk.so: Nectar's Vulkan driver, registered with the Vulkan loader in
#   /usr/share/vulkan/icd.d. It lists no device while nothing is borrowed.
# - /var/lib/nectar: readable by everyone, writable by root only, so no user can point other
#   users' programs at another server. The shared secret in it is readable by root only.
# - The nectar group, for the people allowed to borrow and return (the sudo user is added).
#
# Where NVIDIA's own libcuda.so.1 is installed, Nectar's is left out: CUDA programs keep that GPU,
# and the computer borrows through Vulkan only.
#
#   sudo ./install-borrower.sh --secret "the secret entered on the lender"
#   sudo ./install-borrower.sh --uninstall

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

programs=/opt/nectar
state=/var/lib/nectar
unit=/etc/systemd/system/nectar-borrow.service
library_list=/etc/ld.so.conf.d/00-nectar.conf
vulkan_manifest=/usr/share/vulkan/icd.d/nectar_icd.json

remove() {
    if [ -f "$unit" ]; then
        # Stopping returns a borrowed GPU and removes borrow.cfg
        systemctl disable --now nectar-borrow.service 2>/dev/null || true
        rm -f "$unit"
        systemctl daemon-reload
    fi
    rm -f "$library_list" "$vulkan_manifest" /usr/local/bin/nectar
    ldconfig
    rm -rf "$programs" "$state"
}

if [ "$uninstall" -eq 1 ]; then
    remove
    getent group nectar >/dev/null && groupdel nectar
    echo "Nectar removed"
    exit 0
fi

if [ ${#secret} -lt 8 ]; then
    echo "Give the lender's shared secret, at least 8 characters: --secret <secret>" >&2
    exit 2
fi
for file in nectar nectar-svc nectar-detect cuda-probe libnvcuda.so libnectar_vk.so; do
    if [ ! -f "$source/$file" ]; then
        echo "No $file in $source; give Nectar's Linux build with --source" >&2
        exit 1
    fi
done
# NVIDIA's own driver, anywhere the loader looks, other than a Nectar install
nvidia=$(ldconfig -p | awk '/libcuda\.so\.1 / {print $NF}' | grep -v "^$programs/" || true)

remove
install -d -m 755 "$programs/bin" "$programs/lib" "$state"
for file in nectar nectar-svc nectar-detect cuda-probe; do
    install -m 755 "$source/$file" "$programs/bin/$file"
done
[ -f "$source/nectar-server" ] && install -m 755 "$source/nectar-server" "$programs/bin/"
if [ -n "$nvidia" ]; then
    echo "This computer has NVIDIA's CUDA driver ($nvidia), so it borrows through Vulkan only"
else
    install -m 755 "$source/libnvcuda.so" "$programs/lib/libcuda.so.1"
    ln -sf libcuda.so.1 "$programs/lib/libcuda.so"
    echo "$programs/lib" > "$library_list"
    ldconfig
fi
install -m 755 "$source/libnectar_vk.so" "$programs/lib/libnectar_vk.so"
install -d -m 755 "$(dirname "$vulkan_manifest")"
cat > "$vulkan_manifest" <<EOF
{"file_format_version":"1.0.1","ICD":{"library_path":"$programs/lib/libnectar_vk.so","api_version":"1.4.363"}}
EOF
ln -sf "$programs/bin/nectar" /usr/local/bin/nectar

umask 077
printf '%s' "$secret" > "$state/secret"
umask 022

getent group nectar >/dev/null || groupadd --system nectar
if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
    usermod -aG nectar "$SUDO_USER"
    added=" $SUDO_USER is in the nectar group from the next sign-in."
fi

install -m 644 "$source/nectar-borrow.service" "$unit" 2>/dev/null ||
    install -m 644 "$(dirname "$0")/nectar-borrow.service" "$unit"
systemctl daemon-reload
systemctl enable --now nectar-borrow.service

echo "Nectar installed.${added:-} Then: nectar status, nectar borrow <lender>"
