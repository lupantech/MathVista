import base64
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import AzureOpenAI
from PIL import Image

from models.gpt import GPT_Model


def make_client():
    return AzureOpenAI(
        api_key="test-key",
        azure_endpoint="https://example.invalid",
        api_version="2024-02-01",
    )


class CompatibilityTests(unittest.TestCase):
    def test_default_azure_client_initializes(self):
        with make_client() as client:
            self.assertFalse(client.is_closed())

    def test_evaluation_entrypoint_imports(self):
        importlib.import_module("evaluation.generate_response")

    def test_dataset_image_request_preserves_pixels(self):
        from datasets import Dataset, Features, Image as ImageFeature
        from utilities import get_image_size

        source = Image.new("RGB", (7, 5), (21, 43, 65))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            source.save(path)
            self.assertEqual(get_image_size(path), source.size)
            dataset = Dataset.from_dict(
                {"decoded_image": [{"bytes": path.read_bytes(), "path": None}]},
                features=Features({"decoded_image": ImageFeature()}),
            )
            decoded_image = dataset[0]["decoded_image"]

        requests = []

        def respond(client, request, **kwargs):
            requests.append(request)
            return httpx.Response(200, request=request, json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4-vision-preview",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }],
            })

        with patch("httpx.Client.send", autospec=True, side_effect=respond):
            with make_client() as client:
                model = GPT_Model(client, model="gpt-4-vision-preview", patience=1)
                self.assertEqual(model.get_response("Describe", decoded_image), "ok")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(requests[0].headers["api-key"], "test-key")
        body = json.loads(requests[0].content)
        url = body["messages"][0]["content"][0]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as output:
            self.assertEqual(output.size, source.size)
            self.assertEqual(output.tobytes(), source.tobytes())


if __name__ == "__main__":
    unittest.main()
