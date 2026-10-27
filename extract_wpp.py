#!/usr/bin/env python3
"""HEVC WPP row bytes without entropy decoding. Requires ffmpeg, ffprobe, numpy,
matplotlib. Supports local MP4/MOV with hvcC, one slice/picture, WPP, no tiles.
FFmpeg trace_headers parses syntax; original MP4 sample positions provide exact
NAL lengths. No frame decoding; heuristic decoder probing is explicitly disabled.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np


EXPORT_SCHEMA_VERSION = '2.0'
FRAME_METADATA_KEYS = ('display_index', 'decode_index', 'pts', 'pts_seconds',
                       'slice_type', 'slice_qp', 'temporal_id', 'nal_type',
                       'vcl_nal_bytes', 'header_bytes', 'payload_bytes',
                       'cu_qp_delta_enabled')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probe(path, *args):
    cmd = ['ffprobe', '-v', 'error', '-nofind_stream_info', '-select_streams',
           'v:0', *args, '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(cmd, text=True))


def unhex_dump(text):
    return bytes.fromhex(''.join(line.split(': ', 1)[1].split('  ', 1)[0]
                                for line in text.splitlines() if ': ' in line))


def trace_slices(path):
    """Read trace incrementally (never store the large human-readable trace)."""
    cmd = ['ffmpeg', '-hide_banner', '-nostdin', '-nostats', '-loglevel', 'info',
           '-nofind_stream_info', '-i', str(path), '-map', '0:v:0', '-c:v', 'copy',
           '-bsf:v', 'trace_headers', '-f', 'null', '-']
    pattern = re.compile(r'^\s*(\d+)\s+(\S+)\s+([01]+)\s+=\s+(-?\d+)\s*$')
    sps, pps, result = {}, {}, []
    section, fields, end_bit = None, {}, 0
    tail = deque(maxlen=15)

    def finish():
        if section == 'Sequence Parameter Set':
            sps[fields['sps_seq_parameter_set_id']] = fields.copy()
        elif section == 'Picture Parameter Set':
            pps[fields['pps_pic_parameter_set_id']] = fields.copy()
        elif section == 'Slice Segment Header':
            require('alignment_bit_equal_to_one' in fields and end_bit % 8 == 0,
                    'Incomplete slice header: unsupported FFmpeg trace format or bad stream.')
            p = pps[fields['slice_pic_parameter_set_id']]
            s = sps[p['pps_seq_parameter_set_id']]
            result.append((fields.copy(), end_bit // 8, p.copy(), s.copy()))

    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, encoding='utf-8', errors='replace')
    try:
        for line in proc.stderr:
            tail.append(line.rstrip())
            if '[trace_headers @' not in line:
                continue
            text = line.split(']', 1)[1].strip()
            m = pattern.match(text)
            if m:
                if section:
                    bit, name, binary, value = m.groups()
                    fields[name] = int(value)
                    end_bit = int(bit) + len(binary)
            else:
                finish()
                section = text if text in ('Sequence Parameter Set',
                                           'Picture Parameter Set',
                                           'Slice Segment Header') else None
                fields, end_bit = {}, 0
        finish()
        require(proc.wait() == 0, 'FFmpeg header parsing failed:\n' + '\n'.join(tail))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stderr.close()
    require(result, 'No slice headers found; FFmpeg must include trace_headers.')
    return result


def nal_units(file, packet, length_bytes):
    """MP4 demux packets are contiguous samples at the reported file position."""
    pos, size = int(packet['pos']), int(packet['size'])
    require(pos >= 0 and size > 0, 'Packet lacks a usable local file offset.')
    cursor, end = pos, pos + size
    units = []
    while cursor < end:
        require(cursor + length_bytes <= end, 'Truncated NAL length prefix.')
        file.seek(cursor)
        b = file.read(length_bytes)
        require(len(b) == length_bytes, 'Truncated file.')
        n = int.from_bytes(b, 'big')
        cursor += length_bytes
        require(n >= 2 and cursor + n <= end, 'Invalid MP4 NAL boundary (encrypted or unsupported input?).')
        header = file.read(2)
        require(len(header) == 2 and not (header[0] & 128), 'Invalid HEVC NAL header.')
        typ = (header[0] >> 1) & 63
        if typ < 32:
            units.append((cursor, n, typ))
        cursor += n
    return units


def raw_header_length(file, start, nal_size, rbsp_bytes):
    """Translate unescaped trace bit positions to original NAL byte positions.
    The trace includes the two-byte NAL header; escape removal starts after it.
    """
    require(2 <= rbsp_bytes <= nal_size, 'Invalid slice-header length.')
    file.seek(start + 2)
    raw_count, decoded_count, zeros = 2, 2, 0
    while decoded_count < rbsp_bytes:
        b = file.read(1)
        require(b and raw_count < nal_size, 'Truncated slice header.')
        raw_count += 1
        v = b[0]
        if zeros >= 2 and v == 3:
            zeros = 0
            continue
        decoded_count += 1
        zeros = zeros + 1 if v == 0 else 0
    return raw_count


def extract(path):
    info = probe(path, '-show_streams', '-show_format', '-show_data',
                 '-show_entries', 'stream=codec_name,extradata,time_base:format=format_name')
    require('mov' in info['format']['format_name'].split(','),
            'This extractor supports MP4/MOV only. Remux other containers to MP4 first.')
    require(info.get('streams') and info['streams'][0]['codec_name'] == 'hevc',
            'First video stream must be HEVC/H.265.')
    st = info['streams'][0]
    extra = unhex_dump(st.get('extradata', ''))
    require(len(extra) >= 23 and extra[0] == 1, 'Expected hvcC HEVC extradata.')
    length_bytes = (extra[21] & 3) + 1
    tb = Fraction(st['time_base'])
    print('Parsing HEVC headers (no CABAC decoding)...', file=sys.stderr)
    slices = trace_slices(path)
    packets = probe(path, '-show_packets', '-show_entries', 'packet=pts,dts,pos,size')['packets']
    rows, records, config = [], [], None
    slice_index = 0
    with path.open('rb') as f:
        for packet in packets:
            units = nal_units(f, packet, length_bytes)
            if not units:
                continue
            require(len(units) == 1, 'Multiple VCL NALs in one picture/sample: use slices=1.')
            require(slice_index < len(slices), 'Packet/header count mismatch.')
            h, rbsp_header, p, s = slices[slice_index]
            start, nal_size, nal_type = units[0]
            require(nal_type == h['nal_unit_type'], 'NAL/header ordering mismatch.')
            require(h['nuh_layer_id'] == 0, 'Multilayer HEVC is unsupported.')
            require(h['first_slice_segment_in_pic_flag'] == 1, 'Multiple slices per picture are unsupported.')
            require(p.get('tiles_enabled_flag', 0) == 0, 'Tiles are enabled: whole-width rows cannot be assumed.')
            require(p.get('entropy_coding_sync_enabled_flag', 0) == 1,
                    'WPP is disabled. Encode with wpp=1 and a working thread pool.')
            require(not s.get('separate_colour_plane_flag', 0), 'Separate colour planes unsupported.')
            width, height = s['pic_width_in_luma_samples'], s['pic_height_in_luma_samples']
            ctu = 1 << (s['log2_min_luma_coding_block_size_minus3'] + 3 +
                        s['log2_diff_max_min_luma_coding_block_size'])
            nrows = (height + ctu - 1) // ctu
            geom = (width, height, ctu, s.get('chroma_format_idc', 1),
                    s.get('bit_depth_luma_minus8', 0),
                    *(s.get('conf_win_' + edge + '_offset', 0)
                      for edge in ('left', 'right', 'top', 'bottom')))
            if config is None:
                config = geom
            require(geom == config, 'Geometry/format changed midstream; split the video first.')
            n = h.get('num_entry_point_offsets', 0)
            require(n == nrows - 1,
                    f'Expected {nrows - 1} row offsets, got {n}; incompatible slice layout.')
            raw_header = raw_header_length(f, start, nal_size, rbsp_header)
            payload = nal_size - raw_header
            lengths = [h[f'entry_point_offset_minus1[{i}]'] + 1 for i in range(n)]
            lengths.append(payload - sum(lengths))
            require(all(x > 0 for x in lengths), 'Invalid substream sizes or final subset length.')
            require('pts' in packet, 'Missing presentation timestamps; remux/re-encode with timestamps.')
            pts = int(packet['pts'])
            records.append(dict(decode_index=slice_index, pts=pts,
                                pts_seconds=float(pts * tb),
                                slice_type={0: 'B', 1: 'P', 2: 'I'}[h['slice_type']],
                                slice_qp=26 + p['init_qp_minus26'] + h['slice_qp_delta'],
                                temporal_id=h['nuh_temporal_id_plus1'] - 1,
                                nal_type=nal_type, vcl_nal_bytes=nal_size,
                                header_bytes=raw_header, payload_bytes=payload,
                                cu_qp_delta_enabled=p.get('cu_qp_delta_enabled_flag', 0)))
            rows.append(lengths)
            slice_index += 1
    require(slice_index == len(slices), 'Not all parsed slices matched MP4 samples.')
    require(records, 'No pictures extracted.')
    order = sorted(range(len(records)), key=lambda i: records[i]['pts'])
    require(len({r['pts'] for r in records}) == len(records), 'Duplicate PTS: ambiguous display order.')
    records = [records[i] for i in order]
    for i, record in enumerate(records):
        record['display_index'] = i
    meta = dict(source=str(path), coded_width=config[0], coded_height=config[1],
                ctu_size=config[2], rows=len(rows[0]), frames=len(records),
                chroma_format_idc=config[3], bit_depth_luma=config[4] + 8,
                conformance_window_offsets=list(config[5:]), time_base=str(tb),
                measurement='Original escaped slice-data subset bytes, including termination/alignment and any final padding; excludes NAL/slice headers and length prefixes.',
                qp_note='Initial slice QP only; not mean row QP. Local QP changes are not decoded.',
                method='FFmpeg trace_headers + MP4 packet/NAL boundaries; heuristic decoder probing disabled.')
    return np.asarray(rows, dtype=np.int64)[order], records, meta


def plot_wpp_rows(path, values, meta, types):
    """One bar per selected frame, with spatial rows stacked top to bottom."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nframes, nrows = values.shape
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 2 * nrows + 1),
                             sharex=True, sharey=True, squeeze=False)
    payload_kib = values / 1024
    for row, ax in enumerate(axes[:, 0]):
        ax.bar(np.arange(nframes), payload_kib[:, row], width=1,
               color='#22658a', linewidth=0)
        ax.set_title(f'WPP row {row}', loc='left', fontsize=10)
        ax.set_ylabel('Payload (KiB)')
        ax.grid(axis='both', alpha=0.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelbottom=True)
    axes[0, 0].set_xlim(-0.5, nframes - 0.5)
    axes[0, 0].set_ylim(0, max(float(payload_kib.max()) * 1.05, 0.001))
    axes[-1, 0].set_xlabel('Display frame index' if types == 'all' else
                           f'Selected {types} picture index in display order (time gaps vary)')
    fig.suptitle(f'{Path(meta["source"]).name} • {types} pictures\n'
                 'WPP row payload across frames • top-to-bottom row order • shared KiB scale',
                 y=1 - 0.15 / fig.get_figheight())
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()), h_pad=1.2)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_machine_outputs(out, values, records, meta, types):
    """Write the versioned row-by-presentation-frame producer schema.

    ``values`` remains presentation-frame x WPP-row internally so plotting can use
    it directly. Machine-readable row measurements are transposed at this boundary.
    """
    require(values.ndim == 2, 'Row-byte measurements must be a two-dimensional array.')
    require(len(records) == values.shape[0], 'Frame metadata and row measurements differ in length.')
    require(values.shape[0] > 0 and values.shape[1] > 0, 'No WPP row measurements to export.')
    for record in records:
        require(all(key in record for key in FRAME_METADATA_KEYS),
                'Incomplete per-frame metadata.')

    total = values.sum(axis=1)
    shares = values / total[:, None]
    log_bytes = np.log1p(values)
    c, h, w = meta['ctu_size'], meta['coded_height'], meta['coded_width']
    row_heights = np.minimum(c, h - np.arange(values.shape[1]) * c)
    bpp = 8 * values / (w * row_heights[None, :])
    delta = np.vstack([np.zeros((1, values.shape[1])), np.diff(shares, axis=0)])

    # The producer contract is WPP rows x presentation frames. Do not pad or crop
    # the frame axis: every input picture gets one CSV column and NPZ column.
    row_by_frame = values.T
    with (out / 'rows.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'frame_{i}' for i in range(values.shape[0])])
        writer.writerows(row.tolist() for row in row_by_frame)
    with (out / 'frames.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(FRAME_METADATA_KEYS)
        for record in records:
            writer.writerow([record[key] for key in FRAME_METADATA_KEYS])

    np.savez_compressed(
        out / 'features.npz',
        row_bytes=row_by_frame,
        row_share=shares.T,
        log1p_row_bytes=log_bytes.T,
        row_bits_per_coded_pixel=bpp.T,
        delta_row_share=delta.T,
        total_payload_bytes=total,
        **{key: np.array([record[key] for record in records])
           for key in FRAME_METADATA_KEYS})
    output_meta = dict(meta)
    output_meta.update({
        'schema_version': EXPORT_SCHEMA_VERSION,
        'plot_types': types,
        'npz_array_layout': 'WPP CTU rows (top to bottom) x presentation frames',
        'export_layout': {
            'presentation_frame_order': 'ascending PTS (display order)',
            'wpp_row_order': 'top to bottom CTU row; CSV record number is the zero-based row index',
            'rows_csv': 'WPP CTU rows x presentation frames; columns are frame_0 through frame_(T-1)',
            'frames_csv': 'presentation frames x per-frame metadata; row i describes frame_i in rows.csv and NPZ',
            'features_npz': {
                'row_bytes': 'WPP CTU rows x presentation frames',
                'row_share': 'WPP CTU rows x presentation frames',
                'log1p_row_bytes': 'WPP CTU rows x presentation frames',
                'row_bits_per_coded_pixel': 'WPP CTU rows x presentation frames',
                'delta_row_share': 'WPP CTU rows x presentation frames; delta from previous presentation frame, first column zero',
                'total_payload_bytes': 'presentation frames',
            },
        },
        'frame_metadata_fields': list(FRAME_METADATA_KEYS),
        'delta_note': 'Difference from previous display frame; first difference is zero. Mixed types may confound differences.',
    })
    (out / 'metadata.json').write_text(json.dumps(output_meta, indent=2) + '\n')
    return total, shares, log_bytes, bpp, delta


def save_outputs(out, values, records, meta, types):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    for name in ['rows.csv', 'frames.csv', 'features.npz', 'metadata.json', 'heatmap.png',
                 'wpp_rows.png', 'wpp_rows_P.png', 'wpp_rows_PB.png', 'wpp_rows_I.png']:
        require(not (out / name).exists(), f'{out / name} exists; choose a new --out directory.')
    mask = np.array([types == 'all' or r['slice_type'] in types for r in records])
    require(mask.any(), f'No {types} pictures; use --types all or another type.')
    total, shares, log_bytes, _bpp, _delta = write_machine_outputs(out, values, records, meta, types)
    # Type filtering affects the plot only. Full feature arrays always remain intact.
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 1, 2]}, layout='constrained')
    for ax, matrix, title, label in [
        (axes[0], log_bytes[mask], 'Absolute row payload', 'ln(1 + bytes)'),
        (axes[1], shares[mask], 'Within-picture row share', 'Fraction of payload')]:
        image = ax.imshow(matrix.T, origin='upper', aspect='auto',
                          interpolation='nearest', cmap='magma')
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_ylabel('CTU row (top → bottom)')
        fig.colorbar(image, ax=ax, label=label, pad=0.015)
    axes[2].plot(np.arange(mask.sum()), total[mask] / 1024, lw=0.8, color='#22658a')
    axes[2].set_title('Total frame payload (sum of WPP rows)', loc='left', fontsize=11)
    axes[2].set_ylabel('Payload (KiB)')
    axes[2].grid(alpha=0.2)
    nrows = values.shape[1]
    cmap = plt.get_cmap('turbo', nrows)
    norm = matplotlib.colors.BoundaryNorm(np.arange(nrows + 1) - 0.5, nrows)
    for row in range(nrows):
        axes[3].plot(np.arange(mask.sum()), values[mask, row] / 1024,
                     lw=0.8, color=cmap(norm(row)))
    axes[3].set_title('Payload per WPP row', loc='left', fontsize=11)
    axes[3].set_ylabel('Row payload (KiB)')
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[3],
                 label='CTU row (0 = top)', pad=0.015,
                 ticks=np.unique(np.linspace(0, nrows - 1, min(nrows, 10), dtype=int)))
    axes[3].set_xlabel('Display frame index' if types == 'all' else
                       f'Selected {types} picture index in display order (time gaps vary)')
    axes[3].grid(alpha=0.2)
    fig.suptitle(f'{Path(meta["source"]).name} • {meta["rows"]} CTU rows • {types} pictures')
    fig.savefig(out / 'heatmap.png', dpi=160)
    plt.close(fig)
    plot_wpp_rows(out / 'wpp_rows.png', values[mask], meta, types)
    for selected_types in ('P', 'PB', 'I'):
        selected = np.array([r['slice_type'] in selected_types for r in records])
        if selected.any():
            plot_wpp_rows(out / f'wpp_rows_{selected_types}.png',
                          values[selected], meta, selected_types)
        else:
            print(f'Skipping wpp_rows_{selected_types}.png: no matching pictures.',
                  file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    p.add_argument('--out', type=Path, help='New output directory; default VIDEO_wpp')
    p.add_argument('--types', default='all', choices=['all', 'I', 'P', 'B', 'IP', 'PB'],
                   help='Filter plots only; CSV/NPZ always contain every picture.')
    args = p.parse_args()
    try:
        for executable in ['ffmpeg', 'ffprobe']:
            require(shutil.which(executable), f'Install {executable} and add it to PATH.')
        path = args.video.expanduser().resolve()
        require(path.is_file(), f'Input does not exist: {path}')
        values, records, meta = extract(path)
        out = args.out or path.with_name(path.stem + '_wpp')
        save_outputs(out, values, records, meta, args.types)
        print(f'Saved {len(records)} pictures × {meta["rows"]} rows to {out.resolve()}')
        if any(r['cu_qp_delta_enabled'] for r in records):
            print('Note: CU QP deltas are enabled; slice QP is not actual row QP.', file=sys.stderr)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as e:
        p.exit(1, f'Error: {e}\n')


if __name__ == '__main__':
    main()
