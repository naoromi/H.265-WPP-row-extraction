#!/usr/bin/env bash
# Nonrecursive batch conversion. Usage: bash encode_wpp.sh INPUT_DIR [GOP=16] [QP=28] [BFRAMES=0]
# Outputs INPUT_DIR/hevc-wpp-gop-GOP-qp-QP-b-BFRAMES/*.mp4
set -euo pipefail
input=${1:?Usage: bash encode_wpp.sh INPUT_DIR [GOP=16] [QP=28] [BFRAMES=0]}
gop=${2:-16}
qp=${3:-28}
bframes=${4:-0}
[[ -d "$input" ]] || { echo 'Input directory not found' >&2; exit 1; }
[[ "$gop" =~ ^[1-9][0-9]*$ ]] || { echo 'GOP must be positive' >&2; exit 1; }
[[ "$qp" =~ ^[0-9]+$ && "$qp" -le 51 ]] || { echo 'QP must be 0..51' >&2; exit 1; }
[[ "$bframes" =~ ^[0-9]+$ && "$bframes" -le 16 ]] || { echo 'BFRAMES must be 0..16' >&2; exit 1; }
input=$(cd "$input" && pwd)
out="$input/hevc-wpp-gop-$gop-qp-$qp-b-$bframes"
mkdir -p "$out"
# x265 produces untiled pictures; do not pass an unsupported tiles=0 option.
# Explicit pools avoids pools=none disabling WPP. 4 workers is conservative.
params="wpp=1:pools=4:frame-threads=1:ctu=64:slices=1:qp=$qp:aq-mode=0:cutree=0:ipratio=1:pbratio=1:keyint=$gop:min-keyint=$gop:scenecut=0:open-gop=0:bframes=$bframes:b-adapt=0:repeat-headers=0"
count=0
shopt -s nullglob nocaseglob
for src in "$input"/*; do
    [[ -f "$src" ]] || continue
    case "$src" in
        *.[mM][pP]4|*.[mM][oO][vV]|*.[mM][kK][vV]|*.[aA][vV][iI]|*.[wW][eE][bB][mM]|*.[mM]4[vV]) ;;
        *) continue ;;
    esac
    base=$(basename "$src")
    dest="$out/$base.mp4"
    [[ ! -e "$dest" ]] || { echo "Skipping existing: $dest"; continue; }
    tmp="$out/.$base.partial.mp4"
    echo "Encoding: $base"
    if ! ffmpeg -hide_banner -nostdin -n -i "$src" -map 0:v:0 -an -sn -dn \
        -c:v libx265 -preset medium -pix_fmt yuv420p -fps_mode passthrough \
        -x265-params "$params" -tag:v hvc1 "$tmp"; then
        rm -f "$tmp"
        exit 1
    fi
    # Check the actual first picture's PPS, not just the requested settings.
    check="$out/.$base.check.log"
    if ! ffmpeg -hide_banner -nostdin -nostats -loglevel info -nofind_stream_info \
        -i "$tmp" -map 0:v:0 -c:v copy -frames:v 1 -bsf:v trace_headers -f null - 2> "$check"; then
        echo "Header validation failed: $check" >&2; exit 1
    fi
    if ! grep -Eq 'entropy_coding_sync_enabled_flag.*= 1$' "$check" || \
       grep -Eq 'entropy_coding_sync_enabled_flag.*= 0$|tiles_enabled_flag.*= 1$' "$check"; then
        echo "Not compatible WPP/no-tiles output (possibly too small): $tmp; see $check" >&2
        exit 1
    fi
    rm -f "$check"
    mv "$tmp" "$dest"
    count=$((count + 1))
done
echo "Encoded $count videos into: $out"
