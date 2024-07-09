import io
import os
import shutil
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from phi_3_vision_mlx import (PATH_QUANTIZED_PHI3_VISION, Agent, _setup,
                              benchmark, test_lora, train_lora)

class TestPhi3VisionMLX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _setup()
        cls.agent = Agent(quantize_model=True)
        cls.model_path = PATH_QUANTIZED_PHI3_VISION
        cls.tmp_dir = Path('tmp')
        cls.adapter_path = cls.tmp_dir / cls.model_path
        cls.json_path = cls.tmp_dir / 'benchmark.json'
        cls.tmp_dir.mkdir(exist_ok=True)

        try:
            train_lora(
                model_path=cls.model_path,
                adapter_path=cls.adapter_path,
                lora_layers=2,
                lora_rank=2,
                epochs=2,
                take=4,
                batch_size=2,
                lr=1e-4,
                warmup=.5,
                dataset_path="JosefAlbers/akemiH_MedQA_Reason"
            )
        except Exception as e:
            print(f"LoRA training failed: {str(e)}")
            raise unittest.SkipTest(f"Failed to train LoRA: {str(e)}")
        if not cls.adapter_path.exists():
            raise AssertionError("Adapter files should be created")

    def test_multi_turn_vqa(self):
        response1 = self.agent('What is shown in this image?', 'https://collectionapi.metmuseum.org/api/collection/v1/iiif/344291/725918/main-image')
        self.assertIsNotNone(response1)
        response2 = self.agent('What is the location?')
        self.assertIsNotNone(response2)
        self.agent.end()

    def test_generative_feedback_loop(self):
        response1 = self.agent('Plot a Lissajous Curve.')
        self.assertIsNotNone(response1)
        response2 = self.agent('Modify the code to plot 3:4 frequency')
        self.assertIsNotNone(response2)
        self.agent.end()

    def test_api_tool_use(self):
        response1 = self.agent('Draw "A perfectly red apple, 32k HDR, studio lighting"')
        self.assertIsNotNone(response1)
        self.agent.end()
        response2 = self.agent('Speak "People say nothing is impossible, but I do nothing every day."')
        self.assertIsNotNone(response2)
        self.agent.end()

    def test_benchmark(self):
        try:
            benchmark(json_path=self.json_path)
        except Exception as e:
            self.fail(f"benchmark() raised {type(e).__name__} unexpectedly!")
        self.assertTrue(self.json_path.exists(), "benchmark.json file should be created")

    def test_lora(self):
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                test_lora(
                    model_path=self.model_path,
                    adapter_path=self.adapter_path,
                    take=2
                )
            except Exception as e:
                self.fail(f"test_lora() raised {type(e).__name__} unexpectedly!")
        output = f.getvalue()
        self.assertIn("Score:", output, "Expected 'Score:' in the output")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'agent'):
            del cls.agent

        if hasattr(cls, 'tmp_dir'):
            try:
                shutil.rmtree(cls.tmp_dir)
                print(f"Removed temporary directory: {cls.tmp_dir}")
            except Exception as e:
                print(f"Error removing temporary directory: {e}")

if __name__ == '__main__':
    unittest.main()
