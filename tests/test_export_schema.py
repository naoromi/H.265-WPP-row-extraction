import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import extract_wpp


def records(count):
    """Presentation-ordered records with deliberately non-presentation decode order."""
    types = ('I', 'P', 'B')
    return [
        {
            'display_index': frame,
            'decode_index': (count - frame - 1),
            'pts': 1000 + frame * 40,
            'pts_seconds': frame / 25,
            'slice_type': types[frame % len(types)],
            'slice_qp': 28 + frame,
            'temporal_id': frame % 2,
            'nal_type': 1 + frame,
            'vcl_nal_bytes': 100 + frame,
            'header_bytes': 10,
            'payload_bytes': 90 + frame,
            'cu_qp_delta_enabled': frame % 2,
        }
        for frame in range(count)
    ]


def meta(rows):
    return {
        'source': 'synthetic.mp4',
        'coded_width': 128,
        'coded_height': rows * 64,
        'ctu_size': 64,
        'rows': rows,
        'frames': None,
    }


class ExportSchemaTests(unittest.TestCase):
    def test_csv_npz_metadata_and_plots_use_row_by_presentation_frame_layout(self):
        # Input to save_outputs is its unchanged plotting layout: frames x rows.
        values = np.array([[11, 12], [21, 22], [31, 32]], dtype=np.int64)
        frame_records = records(len(values))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            extract_wpp.save_outputs(out, values, frame_records, meta(values.shape[1]), 'all')

            with (out / 'rows.csv').open(newline='') as file:
                table = list(csv.reader(file))
            self.assertEqual(table[0], ['frame_0', 'frame_1', 'frame_2'])
            np.testing.assert_array_equal(
                np.asarray(table[1:], dtype=np.int64), values.T)

            with np.load(out / 'features.npz') as features:
                np.testing.assert_array_equal(features['row_bytes'], values.T)
                for measurement in ('row_share', 'log1p_row_bytes',
                                    'row_bits_per_coded_pixel', 'delta_row_share'):
                    self.assertEqual(features[measurement].shape, values.T.shape)
                expected_shares = values / values.sum(axis=1, keepdims=True)
                expected_delta = np.vstack([np.zeros((1, values.shape[1])),
                                            np.diff(expected_shares, axis=0)])
                np.testing.assert_allclose(features['row_share'], expected_shares.T)
                np.testing.assert_allclose(features['log1p_row_bytes'], np.log1p(values).T)
                np.testing.assert_allclose(features['row_bits_per_coded_pixel'],
                                           (8 * values / (128 * 64)).T)
                np.testing.assert_allclose(features['delta_row_share'], expected_delta.T)
                np.testing.assert_array_equal(features['row_share'].sum(axis=0), np.ones(3))
                # Frame-vector order remains the supplied presentation order, not decode order.
                for key in extract_wpp.FRAME_METADATA_KEYS:
                    np.testing.assert_array_equal(
                        features[key], np.array([record[key] for record in frame_records]))
                np.testing.assert_array_equal(features['pts'], [1000, 1040, 1080])
                np.testing.assert_array_equal(features['decode_index'], [2, 1, 0])

            with (out / 'frames.csv').open(newline='') as file:
                frame_table = list(csv.DictReader(file))
            self.assertEqual(len(frame_table), 3)
            self.assertEqual(frame_table[0]['pts'], '1000')
            self.assertEqual(frame_table[2]['decode_index'], '0')
            self.assertEqual(set(frame_table[0]), set(extract_wpp.FRAME_METADATA_KEYS))

            metadata = json.loads((out / 'metadata.json').read_text())
            self.assertEqual(metadata['schema_version'], '2.0')
            self.assertEqual(metadata['export_layout']['presentation_frame_order'],
                             'ascending PTS (display order)')
            self.assertIn('WPP CTU rows x presentation frames',
                          metadata['export_layout']['features_npz']['row_bytes'])
            for plot in ('heatmap.png', 'wpp_rows.png', 'wpp_rows_P.png',
                         'wpp_rows_PB.png', 'wpp_rows_I.png'):
                self.assertTrue((out / plot).is_file(), plot)

    def test_extract_sorts_decode_order_by_pts_before_returning_measurements(self):
        # Avoid FFmpeg dependencies while exercising extract's decode-to-presentation
        # reordering, including the matching row-byte measurements.
        pps = {
            'tiles_enabled_flag': 0,
            'entropy_coding_sync_enabled_flag': 1,
            'init_qp_minus26': 0,
            'cu_qp_delta_enabled_flag': 0,
        }
        sps = {
            'pic_width_in_luma_samples': 128,
            'pic_height_in_luma_samples': 128,
            'log2_min_luma_coding_block_size_minus3': 3,
            'log2_diff_max_min_luma_coding_block_size': 0,
        }
        headers = [
            ({'nal_unit_type': 1, 'nuh_layer_id': 0,
              'first_slice_segment_in_pic_flag': 1, 'num_entry_point_offsets': 1,
              'entry_point_offset_minus1[0]': 9, 'slice_type': 1,
              'slice_qp_delta': 0, 'nuh_temporal_id_plus1': 1}, 20, pps, sps),
            ({'nal_unit_type': 1, 'nuh_layer_id': 0,
              'first_slice_segment_in_pic_flag': 1, 'num_entry_point_offsets': 1,
              'entry_point_offset_minus1[0]': 19, 'slice_type': 2,
              'slice_qp_delta': 0, 'nuh_temporal_id_plus1': 1}, 20, pps, sps),
        ]
        stream_info = {
            'format': {'format_name': 'mov,mp4'},
            'streams': [{'codec_name': 'hevc', 'extradata': '', 'time_base': '1/1000'}],
        }
        packets = {'packets': [
            {'pts': '200', 'pos': '0', 'size': '100'},  # decoded first, displayed second
            {'pts': '100', 'pos': '100', 'size': '100'},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / 'synthetic.mp4'
            video.write_bytes(b'')
            with patch.object(extract_wpp, 'probe', side_effect=[stream_info, packets]), \
                    patch.object(extract_wpp, 'unhex_dump', return_value=bytes([1]) + bytes(22)), \
                    patch.object(extract_wpp, 'trace_slices', return_value=headers), \
                    patch.object(extract_wpp, 'nal_units', return_value=[(0, 100, 1)]), \
                    patch.object(extract_wpp, 'raw_header_length', return_value=20):
                values, extracted_records, extracted_meta = extract_wpp.extract(video)

        # PTS 100 was decoded second and has a different entry-point size, proving
        # rows and metadata are reordered together into presentation order.
        np.testing.assert_array_equal(values, [[20, 60], [10, 70]])
        self.assertEqual([record['pts'] for record in extracted_records], [100, 200])
        self.assertEqual([record['decode_index'] for record in extracted_records], [1, 0])
        self.assertEqual([record['display_index'] for record in extracted_records], [0, 1])
        self.assertEqual(extracted_meta['frames'], 2)

    def test_long_and_short_frame_axes_are_never_padded_or_truncated(self):
        for frame_count in (1, 3001):
            values = (np.arange(frame_count * 2, dtype=np.int64).reshape(frame_count, 2)
                      + 1)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                extract_wpp.write_machine_outputs(out, values, records(frame_count),
                                                  meta(values.shape[1]), 'all')
                with (out / 'rows.csv').open(newline='') as file:
                    table = list(csv.reader(file))
                self.assertEqual(len(table[0]), frame_count)
                self.assertEqual(table[0][0], 'frame_0')
                self.assertEqual(table[0][-1], f'frame_{frame_count - 1}')
                self.assertEqual(len(table) - 1, 2)
                np.testing.assert_array_equal(
                    np.asarray(table[1:], dtype=np.int64), values.T)
                with np.load(out / 'features.npz') as features:
                    self.assertEqual(features['row_bytes'].shape, (2, frame_count))
                    np.testing.assert_array_equal(features['row_bytes'], values.T)
                    self.assertEqual(len(features['pts']), frame_count)
                    self.assertEqual(features['pts'][-1], 1000 + (frame_count - 1) * 40)


if __name__ == '__main__':
    unittest.main()
