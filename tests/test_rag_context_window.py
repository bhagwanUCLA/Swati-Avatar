import unittest
from types import SimpleNamespace

from backend.core.rag_query import ConversationTooLongError, RAG
from backend.services.firestore_sessions import InMemorySessionStore


class FakeModels:
    def retrieve(self, model_id):
        return SimpleNamespace(max_input_tokens=7_050, max_tokens=2_048)


class FakeMessages:
    def count_tokens(self, *, messages, **kwargs):
        return SimpleNamespace(input_tokens=len(messages) * 1_000)


class FakeClient:
    models = FakeModels()
    messages = FakeMessages()


class FailingDatabase:
    def search(self, question, **kwargs):
        raise RuntimeError("embedding provider unavailable")


class ContextWindowTests(unittest.TestCase):
    def setUp(self):
        self.rag = RAG(db=None, model="test-model", session_store=InMemorySessionStore())
        self.client = FakeClient()

    def test_oldest_saved_pair_is_pruned_without_mutating_history(self):
        history = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        original_history = list(history)
        transient_messages = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "result"}]},
        ]

        messages, retained_history = self.rag._messages_that_fit(
            self.client,
            history,
            "current question",
            transient_messages,
            self.rag._model_input_limit(self.client),
        )

        self.assertEqual(retained_history, history[2:])
        self.assertEqual(history, original_history)
        self.assertEqual(messages[-2:], transient_messages)

    def test_current_question_without_a_saved_pair_raises_clear_error(self):
        client = FakeClient()
        client.models = SimpleNamespace(
            retrieve=lambda model_id: SimpleNamespace(max_input_tokens=2_500, max_tokens=2_048)
        )

        with self.assertRaises(ConversationTooLongError):
            self.rag._messages_that_fit(
                client,
                [],
                "current question",
                [],
                self.rag._model_input_limit(client),
            )

    def test_search_failure_is_returned_as_a_tool_error(self):
        rag = RAG(db=FailingDatabase(), session_store=InMemorySessionStore())

        content, sources, failed = rag._run_search_tool(
            {"query": "mindfulness"},
            default_top_k=6,
            default_section=None,
            default_doc_type=None,
        )

        self.assertTrue(failed)
        self.assertEqual(sources, [])
        self.assertIn("temporarily unavailable", content)


if __name__ == "__main__":
    unittest.main()
