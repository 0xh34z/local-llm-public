"""Unit tests for outputs/byte_budget.py using a tiny synthetic GGUF.

Run from the repository root with:
    python outputs/test_byte_budget.py

The gguf package is only needed to build the fixture; byte_budget itself reads
whatever gguf-py is available on its --gguf-py path.
"""
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import byte_budget  # noqa: E402


def write_model(path, *, with_output=True):
    import numpy as np
    from gguf import GGUFWriter

    writer = GGUFWriter(str(path), 'qwen35moe')
    writer.add_block_count(3)
    writer.add_nextn_predict_layers(1)
    writer.add_expert_count(8)
    writer.add_expert_used_count(2)
    writer.add_context_length(64)
    writer.add_embedding_length(16)
    writer.add_feed_forward_length(32)
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_key_length(8)
    writer.add_ssm_conv_kernel(3)
    writer.add_ssm_inner_size(8)
    writer.add_ssm_group_count(2)
    writer.add_ssm_state_size(2)
    writer.add_ssm_time_step_rank(4)
    writer.add_vocab_size(4)
    writer.add_tokenizer_model('llama')
    writer.add_token_list(['a', 'b', 'c', 'd'])

    writer.add_tensor('token_embd.weight', np.zeros((4, 16), dtype=np.float32))
    if with_output:
        writer.add_tensor('output.weight', np.zeros((4, 16), dtype=np.float32))

    for layer in range(3):
        prefix = f'blk.{layer}.'
        if layer != 1:  # attention layers: 0 and 2 (MTP)
            writer.add_tensor(prefix + 'attn_q.weight', np.zeros((16, 16), dtype=np.float32))
            writer.add_tensor(prefix + 'attn_k.weight', np.zeros((8, 16), dtype=np.float32))
            writer.add_tensor(prefix + 'attn_v.weight', np.zeros((8, 16), dtype=np.float32))
        writer.add_tensor(prefix + 'ssm_conv.weight', np.zeros((8, 3), dtype=np.float32))
        writer.add_tensor(prefix + 'ffn_gate_exps.weight', np.zeros((8, 4, 8), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class ByteBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gguf
        cls.gguf_py = Path(gguf.__file__).resolve().parent.parent

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def run_budget(self, name, **kwargs):
        model = self.dir / f'{name}.gguf'
        write_model(model, **kwargs)
        tensors, hp, vocab, arch = byte_budget.load(model, self.gguf_py)
        result = byte_budget.budget(
            tensors, hp, vocab, model, arch, 100.0, 'synthetic', [0, 16],
        )
        return result

    def test_metadata_and_layer_split(self):
        result = self.run_budget('full')
        self.assertEqual(result['architecture'], 'qwen35moe')
        self.assertEqual(result['vocab_size'], 4)
        self.assertEqual(result['embedding_row_bytes'], 64)
        self.assertEqual(result['block_count'], 3)
        self.assertEqual(result['nextn_predict_layers'], 1)
        self.assertEqual(result['main_layers'], 2)
        self.assertEqual(result['attention_layers'], 1)
        self.assertEqual(result['deltanet_layers'], 1)
        self.assertEqual(result['expert_count'], 8)
        self.assertEqual(result['expert_used_count'], 2)
        self.assertFalse(result['head_tied_to_embedding'])
        self.assertEqual(result['head_bytes'], 256)

    def test_kv_and_recurrent_bytes(self):
        result = self.run_budget('full')
        self.assertEqual(result['kv_bytes_per_token_main'], 32)
        self.assertEqual(result['kv_bytes_per_token_mtp'], 32)
        self.assertEqual(result['recurrent_state_rw_bytes'], 384)
        self.assertEqual(result['routed_bytes_per_token_main'], 512)

    def test_rows_and_floors(self):
        result = self.run_budget('full')
        self.assertEqual([row['depth'] for row in result['rows']], [0, 16])
        row = result['rows'][1]
        for key in ('main_k1_min_bytes', 'main_k3_independent_bytes', 'draft_bytes',
                    'cycle_mtp2_min_bytes', 'cycle_mtp2_max_bytes', 'main_k1_floor_ms',
                    'main_k3_independent_floor_ms', 'draft_floor_ms',
                    'cycle_mtp2_min_floor_ms', 'cycle_mtp2_max_floor_ms'):
            self.assertIn(key, row)
        self.assertGreater(row['draft_floor_ms'], 0)
        self.assertGreater(row['cycle_mtp2_max_bytes'], row['cycle_mtp2_min_bytes'])
        self.assertAlmostEqual(row['cycle_mtp2_min_floor_ms'],
                               row['cycle_mtp2_min_bytes'] / 100.0 / 1e6)

    def test_tied_output_head(self):
        result = self.run_budget('tied', with_output=False)
        self.assertTrue(result['head_tied_to_embedding'])
        self.assertEqual(result['head_bytes'], result['embedding_row_bytes'] * 4)

    def test_find_gguf_py_explicit(self):
        found = byte_budget.find_gguf_py(self.gguf_py)
        self.assertEqual(found.resolve(), self.gguf_py.resolve())

    def test_parse_depths(self):
        self.assertEqual(byte_budget.parse_depths('0, 16, 32'), [0, 16, 32])
        with self.assertRaises(SystemExit):
            byte_budget.parse_depths('oops')
        with self.assertRaises(SystemExit):
            byte_budget.parse_depths('-1')


if __name__ == '__main__':
    unittest.main(verbosity=2)
