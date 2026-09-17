import unittest
from unittest.mock import patch

from ia_claude.llm.factory import get_embedder


class LlmFactoryTests(unittest.TestCase):
    def tearDown(self):
        get_embedder.cache_clear()

    def test_fastembed_embedder_is_reused(self):
        sentinel = object()

        get_embedder.cache_clear()
        with patch(
            "ia_claude.llm.factory.FastEmbedEmbeddings",
            return_value=sentinel,
        ) as embeddings_class:
            first = get_embedder()
            second = get_embedder()

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        embeddings_class.assert_called_once_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


if __name__ == "__main__":
    unittest.main()
