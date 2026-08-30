from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.embeddings.build_doc import product_embed_text
from src.embeddings.store import text_hash

try:
    import numpy as np  # noqa: F401

    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


class TestProductEmbedText(unittest.TestCase):
    def test_composition_title_categories_description(self) -> None:
        product = {
            "parent_asin": "B1",
            "title": "Columbia Men's Thistletown Park Crew",
            "categories": ["Clothing, Shoes & Jewelry", "Men", "Shirts", "T-Shirts"],
            "store": "Columbia",
            "features": ["67% Polyester, 33% Cotton", "Machine Wash", "Imported"],
            "details": {"Material": "polyester"},
            "description": ["A performance crew built for outdoor activity."],
        }
        doc = product_embed_text(product)
        # Title leads, taxonomy + description present.
        self.assertTrue(doc.startswith("Columbia Men's Thistletown Park Crew"))
        self.assertIn("T-Shirts", doc)
        self.assertIn("performance crew", doc)

    def test_excludes_bm25_owned_fields(self) -> None:
        # store/brand, features and details are BM25's job -- not embedded.
        product = {
            "parent_asin": "B2",
            "title": "Basic Tee",
            "store": "Hanes",
            "categories": ["T-Shirts"],
            "features": ["Soft breathable fabric", "100% Cotton"],
            "details": {"Material": "cotton", "Closure Type": "pullover"},
            "description": ["A soft everyday tee."],
        }
        doc = product_embed_text(product)
        self.assertIn("Basic Tee", doc)
        self.assertIn("T-Shirts", doc)
        self.assertIn("everyday tee", doc)
        # Excluded signal:
        self.assertNotIn("Hanes", doc)
        self.assertNotIn("Brand:", doc)
        self.assertNotIn("breathable", doc)
        self.assertNotIn("Closure", doc)
        self.assertNotIn("Material:", doc)

    def test_slices_description_to_budget(self) -> None:
        from src.embeddings.build_doc import _DESCRIPTION_CHAR_BUDGET

        product = {
            "parent_asin": "B7",
            "title": "Jacket",
            "description": ["A" * 1000],
        }
        doc = product_embed_text(product)
        # Only the leading slice of the description is kept.
        self.assertNotIn("A" * (_DESCRIPTION_CHAR_BUDGET + 1), doc)

    def test_falls_back_to_title_categories_without_description(self) -> None:
        product = {
            "parent_asin": "B8",
            "title": "Wool Beanie",
            "categories": ["Hats & Caps", "Beanies"],
            "features": ["100% Wool"],
            "details": {"Material": "wool"},
        }
        doc = product_embed_text(product)
        self.assertEqual(doc, "Wool Beanie Hats & Caps Beanies")

    def test_handles_missing_and_empty_fields(self) -> None:
        self.assertEqual(product_embed_text({"parent_asin": "B3"}), "")
        doc = product_embed_text({"parent_asin": "B4", "title": "  Hat  "})
        self.assertEqual(doc, "Hat")

    def test_respects_overall_char_budget(self) -> None:
        from src.embeddings.build_doc import _DOC_CHAR_BUDGET

        product = {
            "parent_asin": "B5",
            "title": "Jacket " * 200,
            "description": ["x " * 2000],
        }
        self.assertLessEqual(len(product_embed_text(product)), _DOC_CHAR_BUDGET)

    def test_text_hash_is_stable_and_content_sensitive(self) -> None:
        self.assertEqual(text_hash("hello"), text_hash("hello"))
        self.assertNotEqual(text_hash("hello"), text_hash("world"))


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestVectorIndex(unittest.TestCase):
    def _index(self):
        from src.embeddings.index import VectorIndex

        matrix = np.array(
            [
                [1.0, 0.0, 0.0],  # A
                [0.0, 1.0, 0.0],  # B
                [0.9, 0.1, 0.0],  # C (close to A)
            ],
            dtype=np.float32,
        )
        return VectorIndex(["A", "B", "C"], matrix)

    def test_search_orders_by_cosine(self) -> None:
        index = self._index()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hits = index.search(query, top_k=3)
        asins = [asin for asin, _ in hits]
        self.assertEqual(asins[0], "A")
        self.assertEqual(asins[1], "C")
        self.assertEqual(asins[2], "B")
        # Scores are descending and cosine-bounded.
        scores = [score for _, score in hits]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))
        self.assertLessEqual(scores[0], 1.0 + 1e-5)

    def test_top_k_truncates(self) -> None:
        index = self._index()
        hits = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], "A")

    def test_zero_query_returns_empty(self) -> None:
        index = self._index()
        self.assertEqual(index.search(np.zeros(3, dtype=np.float32), top_k=3), [])

    def test_normalization_makes_magnitude_irrelevant(self) -> None:
        index = self._index()
        small = index.search(np.array([0.01, 0.0, 0.0], dtype=np.float32), top_k=1)
        big = index.search(np.array([100.0, 0.0, 0.0], dtype=np.float32), top_k=1)
        self.assertEqual(small[0][0], big[0][0])
        self.assertAlmostEqual(small[0][1], big[0][1], places=4)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class TestEmbeddingCacheRoundTrip(unittest.TestCase):
    def test_save_then_load(self) -> None:
        from src.embeddings.index import VectorIndex
        from src.embeddings.store import load_cache, save_cache

        directory = Path(tempfile.mkdtemp())
        cache_path = directory / "embeddings.npz"
        asins = ["B1", "B2"]
        hashes = [text_hash("doc1"), text_hash("doc2")]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        save_cache(cache_path, model="test-model", asins=asins, hashes=hashes, vectors=vectors)

        loaded = load_cache(cache_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.model, "test-model")
        self.assertEqual(loaded.asins, asins)
        self.assertEqual(loaded.hashes, hashes)
        self.assertEqual(loaded.dim, 2)

        index = VectorIndex.load(cache_path)
        self.assertIsNotNone(index)
        self.assertEqual(len(index), 2)

    def test_load_missing_returns_none(self) -> None:
        from src.embeddings.store import load_cache

        directory = Path(tempfile.mkdtemp())
        self.assertIsNone(load_cache(directory / "nope.npz"))


class TestEmbeddingClientPrefixes(unittest.TestCase):
    """Prefix selection/application (no network; we stub _embed_batch)."""

    def _client(self, **kwargs):
        from src.embeddings.client import EmbeddingClient

        client = EmbeddingClient(
            base_url="http://localhost:1/v1",
            api_key="none",
            model="test-model",
            **kwargs,
        )
        # Force-enable without importing openai / hitting the network, and
        # capture the exact strings that would be sent to the API.
        client._available = True
        self.sent: list[str] = []

        def fake_embed_batch(batch):
            self.sent.extend(batch)
            return [[0.0] for _ in batch]

        client._embed_batch = fake_embed_batch  # type: ignore[assignment]
        return client

    def test_default_prefixes_are_empty(self) -> None:
        client = self._client()
        self.assertEqual(client.document_prefix, "")
        self.assertEqual(client.query_prefix, "")
        client.embed(["hello"], kind="document")
        client.embed(["world"], kind="query")
        self.assertEqual(self.sent, ["hello", "world"])

    def test_embeddinggemma_style_prefixes(self) -> None:
        client = self._client(
            document_prefix="title: none | text: ",
            query_prefix="task: search result | query: ",
        )
        client.embed(["a shirt"], kind="document")
        client.embed(["blue shirt"], kind="query")
        self.assertEqual(
            self.sent,
            ["title: none | text: a shirt", "task: search result | query: blue shirt"],
        )

    def test_embed_one_defaults_to_query_prefix(self) -> None:
        client = self._client(query_prefix="Q:", document_prefix="D:")
        client.embed_one("hat")
        self.assertEqual(self.sent, ["Q:hat"])

    def test_unknown_kind_raises(self) -> None:
        client = self._client()
        with self.assertRaises(ValueError):
            client.embed(["x"], kind="bogus")

    def test_truncates_oversized_input_after_prefix(self) -> None:
        client = self._client(document_prefix="P:", max_input_chars=10)
        client.embed(["x" * 100], kind="document")
        self.assertEqual(len(self.sent[0]), 10)
        # Prefix is preserved within the cap (not truncated away).
        self.assertTrue(self.sent[0].startswith("P:"))

    def test_no_truncation_when_disabled(self) -> None:
        client = self._client(max_input_chars=0)
        long = "y" * 5000
        client.embed([long], kind="document")
        self.assertEqual(self.sent[0], long)

    def test_short_input_is_untouched(self) -> None:
        client = self._client(max_input_chars=100)
        client.embed(["short doc"], kind="document")
        self.assertEqual(self.sent[0], "short doc")


class TestEmbeddingOverflowRecovery(unittest.TestCase):
    """_embed_batch must recover from token-window overflow without crashing."""

    def _client(self, token_limit_chars: int):
        """Client whose stubbed raw call 'overflows' for any single input
        longer than ``token_limit_chars`` (simulating the 512-token server
        limit), and returns a fixed-dim vector otherwise. Records API calls."""
        from src.embeddings.client import EmbeddingClient

        client = EmbeddingClient(
            base_url="http://localhost:1/v1", api_key="none", model="m", max_input_chars=0
        )
        client._available = True
        client._last_dim = 3
        self.raw_calls: list[list[str]] = []

        class _OverflowError(Exception):
            pass

        def fake_raw(batch):
            self.raw_calls.append(list(batch))
            # The server rejects a *request* if any single input is too long.
            if any(len(t) > token_limit_chars for t in batch):
                raise _OverflowError("input (999 tokens) is too large to process")
            return [[1.0, 0.0, 0.0] for _ in batch]

        client._embed_raw = fake_raw  # type: ignore[assignment]
        return client

    def test_splits_batch_to_isolate_oversized_item(self) -> None:
        client = self._client(token_limit_chars=10)
        batch = ["ok1", "ok2", "X" * 50, "ok3"]  # one oversized item
        out = client._embed_batch(batch)
        # All four still get a vector (offending one truncated to fit).
        self.assertEqual(len(out), 4)
        self.assertTrue(all(len(v) == 3 for v in out))

    def test_progressively_truncates_single_oversized_item(self) -> None:
        client = self._client(token_limit_chars=20)
        out = client._embed_batch(["Z" * 200])
        self.assertEqual(len(out), 1)
        # The final successful raw call was a truncated version <= limit.
        last = self.raw_calls[-1][0]
        self.assertLessEqual(len(last), 20)

    def test_non_overflow_error_propagates(self) -> None:
        from src.embeddings.client import EmbeddingClient

        client = EmbeddingClient(
            base_url="http://localhost:1/v1", api_key="none", model="m", max_input_chars=0
        )
        client._available = True

        def boom(batch):
            raise RuntimeError("connection refused")

        client._embed_raw = boom  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            client._embed_batch(["anything"])

    def test_unfittable_item_after_first_success_becomes_zero_vector(self) -> None:
        # token limit 0 => nothing fits; but dim is known, so we emit a zero row
        # rather than aborting the build.
        client = self._client(token_limit_chars=0)
        client._last_dim = 3
        out = client._embed_batch(["anything"])
        self.assertEqual(out, [[0.0, 0.0, 0.0]])


class TestRetrieverVectorFallback(unittest.TestCase):
    """retrieve_vector must no-op gracefully when the vector layer is absent."""

    def test_no_vectors_returns_empty(self) -> None:
        import json

        from src.catalog import Catalog
        from src.retrieval import Retriever

        directory = Path(tempfile.mkdtemp())
        catalog_path = directory / "catalog.jsonl"
        catalog_path.write_text(
            json.dumps(
                {
                    "parent_asin": "B1",
                    "title": "Blue cotton shirt",
                    "categories": ["Shirts"],
                    "features": [],
                    "details": {},
                    "store": "Acme",
                    "description": [],
                    "price": 10.0,
                    "average_rating": 4.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        catalog = Catalog(catalog_path)
        retriever = Retriever(catalog)  # no vector index
        self.assertFalse(retriever.has_vectors)
        self.assertEqual(retriever.retrieve_vector("blue shirt", top_k=5), [])
        # BM25 path still works.
        self.assertEqual(retriever.retrieve_bm25({"keywords": ["shirt"]}, top_k=5), ["B1"])


if __name__ == "__main__":
    unittest.main()
