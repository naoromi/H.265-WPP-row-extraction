# HEVC WPP row feature toolkit

Extract a CTU-row × presentation-frame byte matrix and plot its evolution without CABAC entropy decoding or pixel reconstruction.

## Install

Requires Python 3.9+, NumPy, Matplotlib, and FFmpeg/ffprobe with `libx265` and the `trace_headers` bitstream filter.

```bash
# macOS, if needed:
brew install ffmpeg
python3 -m pip install numpy matplotlib
```

## 1. Encode a directory

```bash
bash encode_wpp.sh /path/to/videos 16 28 0
# Arguments: input directory, GOP length, QP, number of B-frames
```

This processes video files immediately inside the directory (not recursively), keeps original dimensions/timestamps, removes audio, and writes to `hevc-wpp-gop-16-qp-28-b-0/`. Output names retain the input extension before `.mp4` to avoid collisions. Existing outputs are skipped. Paths containing spaces work; compatible with macOS Bash 3.

The initial configuration is I/P-only, constant QP, WPP enabled, one slice per picture, 64-pixel CTUs, no tiles, no AQ/CU-tree, and equal I/P/B QP ratios. For a B-frame experiment:

```bash
bash encode_wpp.sh /path/to/videos 16 28 3
```

The underlying command for one input is:

```bash
ffmpeg -hide_banner -nostdin -n -i input.mp4 \
  -map 0:v:0 -an -sn -dn -c:v libx265 -preset medium \
  -pix_fmt yuv420p -fps_mode passthrough \
  -x265-params 'wpp=1:pools=4:frame-threads=1:ctu=64:slices=1:qp=28:aq-mode=0:cutree=0:ipratio=1:pbratio=1:keyint=16:min-keyint=16:scenecut=0:open-gop=0:bframes=0:b-adapt=0:repeat-headers=0' \
  -tag:v hvc1 output.mp4
```

**No tiles:** x265 produces untiled pictures; do not add an unsupported `tiles=0` parameter. The batch script checks the output PPS for WPP enabled and tiles disabled. Very small pictures can cause x265 to disable WPP; these fail validation. `pools=4` is a conservative explicit worker-pool setting, not a quality parameter.

This uses 8-bit YUV 4:2:0 for a controlled baseline. High-bit-depth/HDR inputs need a deliberate separate experiment; this conversion does not tone-map HDR. Different input resolutions still produce different row counts. Standardize geometry separately if your model requires a fixed matrix height.

## 2. Extract and plot

```bash
python3 extract_wpp.py output.mp4 --out features
python3 extract_wpp.py output.mp4 --out features_P --types P
```

`--types` accepts `all`, `I`, `P`, `B`, `IP`, or `PB`. It filters `heatmap.png` and `wpp_rows.png`; CSV/NPZ retain all pictures. Separate P, P+B, and I row figures always use their named frame types, independently of `--types`; empty groups are skipped. Choose a fresh output directory for each run.

For an entire encoded directory:

```bash
for video in /path/to/videos/hevc-wpp-gop-16-qp-28-b-0/*.mp4; do
  python3 extract_wpp.py "$video" --out "${video%.mp4}_features"
done
```

| Output | Contents |
| --- | --- |
| `heatmap.png` | Log row bytes, within-picture row shares, total frame payload, and one payload line per WPP row (KiB) over display frames |
| `wpp_rows.png` | Tall figure with a separate bar plot for every WPP row, stacked from top to bottom; each bar is a frame's row payload, with shared frame and KiB scales for comparison |
| `wpp_rows_P.png`, `wpp_rows_PB.png`, `wpp_rows_I.png` | Same stacked row plots for P only, P and B together, and I only, in display order; each figure shares its own KiB scale across rows |
| `rows.csv` | Numeric WPP-row × presentation-frame byte matrix. Columns are `frame_0` through `frame_(T-1)`; the zero-based data-record index (after the header) is the top-to-bottom CTU-row index. |
| `frames.csv` | One presentation frame per row with complete per-frame metadata: display/decode index, PTS, I/P/B, initial slice QP, temporal ID, NAL/header/payload sizes, and CU-QP flag. Row *i* describes `frame_i` in `rows.csv` and `features.npz`. |
| `features.npz` | Full WPP-row × presentation-frame row-byte/share/log/bits-per-coded-pixel arrays, temporal share differences, totals, and per-frame metadata vectors. |
| `metadata.json` | Coded geometry, measurement definition, export schema version, and explicit CSV/NPZ axis semantics. |

```python
import numpy as np
f = np.load('features/features.npz')
X = f['row_bytes']   # shape: CTU rows × presentation frames
P = f['row_share']   # each frame column sums to 1
p_only = X[:, f['slice_type'] == 'P']
```

### Export schema and compatibility

Exports use schema **2.0**. `row_bytes` and every other two-dimensional row measurement are **WPP CTU rows (top to bottom) × presentation frames (ascending PTS)** in both `rows.csv` and `features.npz`. The exporter preserves the complete observed frame count *T*: it never pads, truncates, or partitions the frame axis. Per-frame metadata is intentionally separate in `frames.csv` (and remains as vectors in the NPZ), so numeric CSV columns map directly to frame order.

This is a breaking change from older checked-in/sample outputs, where `rows.csv` mixed metadata with one picture per record and NPZ matrices were presentation frames × CTU rows. Read `metadata.json`'s `schema_version` and `export_layout` rather than assuming orientation. **If `schema_version` is absent, treat the output as pre-2.0 legacy frame × row data.** The checked-in `features/` directory is legacy and must not be read with the v2 examples above; regenerate it into a fresh directory to obtain schema 2.0. Plot files retain the same visual presentation-frame-over-time behavior; the extractor keeps that layout internally and transposes only at the machine-readable export boundary.

## Meaning and scope

Brighter cells mean more bytes spent on that horizontal band. The representation measures **coding demand**: texture, prediction error, motion syntax, partitioning, and coding overhead are mixed together. It is not optical flow or directly measured Shannon entropy. Within-frame normalization removes overall payload scale, not all QP effects. Initial slice QP is saved as a conditioning feature; no speculative QP correction is applied.

- Supports local, unencrypted HEVC **MP4/MOV**, with hvcC configuration, one slice per picture, one picture per VCL-bearing sample, WPP enabled, no tiles, stable geometry, and valid PTS. Rejects incompatible layouts instead of mislabelling substreams as rows.
- B-frames are sorted by original presentation timestamps. Type-filtered plots use selected-picture index; unequal time gaps remain available in the timestamps. Full-array temporal differences compare consecutive display pictures, even when their types differ.
- FFmpeg `trace_headers` parses parameter sets and slice headers under `-c:v copy`. Both ffmpeg and ffprobe disable heuristic decoder probing via `-nofind_stream_info`.
- MP4 sample positions and length-prefixed NAL boundaries give exact original NAL lengths. Slice-header RBSP byte positions are translated back to original escaped byte positions. Entry-point lengths retain emulation-prevention bytes, as specified by HEVC.
- The final row is the remaining slice-data subset. All row sizes include their coding termination/alignment overhead, and the final subset includes any trailing padding inside the NAL. These are practical subset byte counts, not perfectly isolated arithmetic-coded symbol costs.
- This is an exploration/reference extractor: text tracing adds overhead, and feature/header/packet metadata memory grows with clip length. Benchmark it before making throughput claims. A production implementation should use a native header parser.

## Validation performed

Tested with FFmpeg 6.1.1 / libx265 3.5 on a synthetic vertically moving textured band: 96 pictures, 6 CTU rows, I/P/B reordering, and constant slice QP 28. Verified byte sums, share sums, header/payload accounting, emulation-prevention coordinate conversion, P-only plotting, and rejection of WPP-disabled and multi-slice streams. A second 320×240 clip checked partial bottom-row geometry. Not tested against all FFmpeg builds or arbitrary camera bitstreams.

`demo_heatmap.png` shows the synthetic moving band; it illustrates spatial sensitivity, not semantic recognition accuracy.

## References

- [FFmpeg bitstream filters / trace_headers](https://ffmpeg.org/ffmpeg-bitstream-filters.html#trace_005fheaders)
- [x265 encoder options](https://x265.readthedocs.io/en/stable/cli.html)
- [x265 WPP description](https://x265.readthedocs.io/en/stable/threading.html)
- [ITU-T H.265](https://www.itu.int/rec/T-REC-H.265), slice entry-point semantics.
