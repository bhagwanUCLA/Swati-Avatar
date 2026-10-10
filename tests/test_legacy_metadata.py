import io
import pickle
import sys
import types
import unittest

from backend.core.chunker import DocumentChunk
from backend.core.database import _load_metadata_pickle


class LegacyMetadataTests(unittest.TestCase):
    def test_loads_document_chunks_pickled_before_module_reorganization(self):
        previous_module = sys.modules.get("chunker")
        legacy_module = types.ModuleType("chunker")
        exec(
            """
class DocumentChunk:
    def __init__(self, **fields):
        self.__dict__.update(fields)
""",
            legacy_module.__dict__,
        )
        sys.modules["chunker"] = legacy_module

        try:
            legacy_chunk = legacy_module.DocumentChunk(
                chunk_id="legacy-chunk",
                doc_index=1,
                doc_title="Legacy document",
                section="general",
                doc_url="legacy://document",
                doc_type="text",
                chunk_index=0,
                text="legacy text",
                raw_content="legacy text",
                metadata={},
            )
            payload = pickle.dumps({"meta": {7: legacy_chunk}, "next_id": 8})
        finally:
            if previous_module is None:
                del sys.modules["chunker"]
            else:
                sys.modules["chunker"] = previous_module

        state = _load_metadata_pickle(io.BytesIO(payload))

        self.assertIsInstance(state["meta"][7], DocumentChunk)
        self.assertEqual(state["meta"][7].chunk_id, "legacy-chunk")
        self.assertEqual(state["next_id"], 8)
