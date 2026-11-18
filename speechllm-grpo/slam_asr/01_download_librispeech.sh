#!/usr/bin/env bash
# Download and verify LibriSpeech archives; run after mounting Google Drive.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo "Usage: bash 01_download_librispeech.sh [train-clean-100 train-clean-360 train-other-500 dev-other test-clean test-other]"
  exit 0
fi
mountpoint -q /content/drive || { echo "Önce Google Drive'ı bağlayın." >&2; exit 1; }
archives=/content/drive/MyDrive/slam_asr/data/archives
base=https://openslr.trmal.net/resources/12
mkdir -p "$archives"
cd "$archives"
splits=(train-clean-100 train-clean-360 train-other-500 dev-other test-clean test-other)
if (( $# )); then splits=("$@"); fi
for split in "${splits[@]}"; do
  case "$split" in
    train-clean-100|train-clean-360|train-other-500|dev-other|test-clean|test-other) ;;
    *) echo "Bilinmeyen bölüm: $split" >&2; exit 2 ;;
  esac
done
df -h .
curl -fsSL --retry 4 "$base/md5sum.txt" -o md5sum.txt.tmp
mv md5sum.txt.tmp md5sum.txt
for split in "${splits[@]}"; do
  file="$split.tar.gz"
  digest=$(awk -v f="$file" '{name=$2; sub(/^\*/, "", name); sub(/^\.\//, "", name); if(name==f) print $1}' md5sum.txt)
  [[ "$digest" =~ ^[[:xdigit:]]{32}$ ]] || { echo "MD5 kaydı bulunamadı: $file" >&2; exit 1; }
  verify() { printf '%s  %s\n' "$digest" "$1" | md5sum --check --status; }
  if [[ -f "$file" ]] && verify "$file"; then
    echo "Zaten doğrulanmış: $file"
    continue
  fi
  echo "İndiriliyor: $file"
  if [[ ! -f "$file.part" ]] || ! verify "$file.part"; then
    curl -fL --retry 5 --continue-at - "$base/$file" -o "$file.part"
  fi
  if ! verify "$file.part"; then
    mv "$file.part" "$file.invalid.$(date +%s)"
    echo "MD5 uyuşmadı: $file. Betiği yeniden çalıştırın." >&2
    exit 1
  fi
  mv -f "$file.part" "$file"
  echo "Doğrulandı: $file"
done
echo "Seçilen arşivler hazır: $archives"
