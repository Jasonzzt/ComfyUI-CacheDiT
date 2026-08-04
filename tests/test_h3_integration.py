import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import utils


class _InPlaceBlock(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states.add_(1)


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_InPlaceBlock() for _ in range(4)])


class H3IntegrationTests(unittest.TestCase):
    def test_h3_preset_uses_pattern_3(self):
        preset = utils.get_preset("MiniMax-H3")
        self.assertEqual(preset.forward_pattern, "Pattern_3")
        self.assertEqual((preset.fn_blocks, preset.bn_blocks), (8, 0))
        self.assertFalse(preset.enable_separate_cfg)

    def test_adapter_resolves_blocks_name_and_clones_only_boundaries(self):
        transformer = _Transformer()
        adapter_blocks, blocks_name = utils.prepare_block_adapter_blocks(
            transformer,
            list(transformer.blocks),
            clone_input_indices=[0, 2],
        )

        self.assertEqual(blocks_name, "blocks")
        self.assertIs(adapter_blocks[1], transformer.blocks[1])
        self.assertIs(adapter_blocks[3], transformer.blocks[3])

        original = torch.zeros(1)
        result = adapter_blocks[0](original)
        self.assertEqual(original.item(), 0)
        self.assertEqual(result.item(), 1)

    def test_summary_normalizes_cache_dit_list_result(self):
        stats = types.SimpleNamespace(
            accumulated_cached_steps=9,
            accumulated_executed_steps=20,
            cached_steps=[3, 5, 7, 8, 10, 11, 13, 14, 16],
            residual_diffs={"blocks": [0.05, 0.12, 0.24]},
        )
        fake_cache_dit = types.SimpleNamespace(summary=lambda transformer: [stats, types.SimpleNamespace()])
        previous = sys.modules.get("cache_dit")
        sys.modules["cache_dit"] = fake_cache_dit
        try:
            result = utils.get_summary_stats(object(), num_steps=20)
        finally:
            if previous is None:
                sys.modules.pop("cache_dit", None)
            else:
                sys.modules["cache_dit"] = previous

        self.assertEqual(result["total_steps"], 20)
        self.assertEqual(result["cached_steps"], 9)
        self.assertEqual(result["computed_steps"], 11)
        self.assertAlmostEqual(result["avg_residual_diff"], 0.1366666667)
        self.assertEqual(result["max_residual_diff"], 0.24)


if __name__ == "__main__":
    unittest.main()
