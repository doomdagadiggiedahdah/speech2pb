#!/usr/bin/env bash
# Install cuDNN 9 for CUDA 12 on Ubuntu 24.04
# Usage: sudo bash install_cudnn9.sh
set -e

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo bash $0"
    exit 1
fi

DEB="cudnn-local-repo-ubuntu2404-9.6.0_1.0-1_amd64.deb"
URL="https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/$DEB"

cd /tmp

if [[ ! -f "$DEB" ]]; then
    echo "Downloading cuDNN 9.6.0 (~1-2GB)..."
    wget -q --show-progress "$URL"
else
    echo "Using cached $DEB"
fi

echo "Installing local repo..."
dpkg -i "$DEB"
cp /var/cudnn-local-repo-ubuntu2404-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/

echo "Updating apt and installing cudnn9-cuda-12..."
apt-get update -qq
apt-get install -y cudnn9-cuda-12

echo "Verifying..."
ldconfig
ldconfig -p | grep cudnn_ops | head -2

echo "Done! cuDNN 9 installed."
