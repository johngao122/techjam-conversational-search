from __future__ import annotations

import unittest
from unittest import mock

from src.config import AgentConfig
from src.intent_router import router


class AgentConfigDefaultsTest(unittest.TestCase):
    def test_defaults_are_the_shipped_configuration(self) -> None:
        c = AgentConfig()
        self.assertEqual(c.parser, "rule")
        self.assertEqual(c.retrieval_mode, "bucket")
        self.assertEqual(c.ask_policy, "always_ask")
        self.assertTrue(c.exposure_gate)
        self.assertEqual(c.release_turn, 3)
        self.assertEqual(c.override_policy, "evict_on_conflict")
        self.assertFalse(c.idf_weight)
        self.assertEqual(c.theta, 0.5)

    def test_invalid_values_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AgentConfig(parser="gpt")
        with self.assertRaises(ValueError):
            AgentConfig(retrieval_mode="dense")
        with self.assertRaises(ValueError):
            AgentConfig(release_turn=0)

    def test_replace_only_overrides_non_none(self) -> None:
        base = AgentConfig(theta=0.4)
        out = base.replace(retrieval_mode="legacy", ask_policy=None)
        self.assertEqual(out.retrieval_mode, "legacy")
        self.assertEqual(out.ask_policy, "always_ask")
        self.assertEqual(out.theta, 0.4)


class AgentConfigFromEnvTest(unittest.TestCase):
    def test_empty_environment_reproduces_defaults(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(AgentConfig.from_env(), AgentConfig())

    def test_reads_every_var(self) -> None:
        env = {
            "PARSER_MODE": "llm",
            "RETRIEVAL_MODE": "legacy",
            "ASK_POLICY": "attribute_cycle",
            "EXPOSURE_GATE": "0",
            "RELEASE_TURN": "5",
            "OVERRIDE_POLICY": "keep",
            "IDF_WEIGHT": "1",
            "THETA": "0.7",
        }
        with mock.patch.dict("os.environ", env, clear=True):
            c = AgentConfig.from_env()
        self.assertEqual(
            c,
            AgentConfig(
                parser="llm",
                retrieval_mode="legacy",
                ask_policy="attribute_cycle",
                exposure_gate=False,
                release_turn=5,
                override_policy="keep",
                idf_weight=True,
                theta=0.7,
            ),
        )

    def test_malformed_numbers_fall_back_to_defaults(self) -> None:
        with mock.patch.dict("os.environ", {"THETA": "abc", "RELEASE_TURN": ""}, clear=True):
            c = AgentConfig.from_env()
        self.assertEqual(c.theta, 0.5)
        self.assertEqual(c.release_turn, 3)


class _FakeParser:
    """Stands in for MessageParser / LLMMessageParser without touching a model."""

    def __init__(self, known_categories=None, known_brands=None) -> None:
        self.known_categories = known_categories or set()

    def parse(self, text: str):  # pragma: no cover - not exercised here
        raise AssertionError("not used")


class WarmParserKindSwitchTest(unittest.TestCase):
    def setUp(self) -> None:
        # Reset the module-level caches so each test starts clean.
        router._vocab = None
        router._vocab_path = None
        router._parser = None
        router._parser_kind = None
        self._orig_classes = dict(router._PARSER_CLASSES)

    def tearDown(self) -> None:
        router._PARSER_CLASSES.clear()
        router._PARSER_CLASSES.update(self._orig_classes)
        router._vocab = router._vocab_path = router._parser = router._parser_kind = None

    def test_kind_switch_rebuilds_parser_but_reuses_vocab(self) -> None:
        vocab_calls = mock.Mock(return_value=({"boots"}, {"nike"}))
        rule_cls, llm_cls = mock.Mock(side_effect=_FakeParser), mock.Mock(side_effect=_FakeParser)
        router._PARSER_CLASSES["rule"] = rule_cls
        router._PARSER_CLASSES["llm"] = llm_cls

        with mock.patch.object(router, "load_catalog_vocab", vocab_calls):
            p_rule = router.warm_parser("data/catalog.jsonl", kind="rule")
            router.warm_parser("data/catalog.jsonl", kind="rule")  # cached, no rebuild
            p_llm = router.warm_parser("data/catalog.jsonl", kind="llm")
            # kind=None keeps whatever was last built (llm), does not revert to rule
            p_default = router.warm_parser("data/catalog.jsonl")

        self.assertEqual(vocab_calls.call_count, 1)   # vocab scanned once
        self.assertEqual(rule_cls.call_count, 1)
        self.assertEqual(llm_cls.call_count, 1)
        self.assertIsNot(p_rule, p_llm)
        self.assertIs(p_llm, p_default)
        self.assertEqual(router._parser_kind, "llm")

    def test_unknown_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            router.warm_parser("data/catalog.jsonl", kind="bogus")


class AgentConfigWiringTest(unittest.TestCase):
    """The config actually reaches the pipeline (real catalog, built once)."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.agent import Agent

        cls.agent = Agent(config=AgentConfig(retrieval_mode="legacy", exposure_gate=False, theta=0.3))

    def test_agent_honours_config(self) -> None:
        self.assertEqual(self.agent._mode, "legacy")
        self.assertEqual(self.agent._theta, 0.3)
        self.assertFalse(self.agent._config.exposure_gate)

    def test_respond_returns_contract_shape(self) -> None:
        self.agent.reset("cfg-smoke", {"preference_tags": [], "rating_style": "mixed"})
        out = self.agent.respond("cfg-smoke", "I'm looking for running shoes", 1, 10)
        self.assertIsInstance(out["message"], str)
        self.assertIn(out["ask_attribute"], {None, "category", "material", "color", "size",
                                             "style", "brand", "budget", "feature", "use_case", "other"})
        self.assertIsInstance(out["recommendations"], list)

    def test_explicit_theta_arg_overrides_config(self) -> None:
        from src.agent import Agent

        a = Agent(config=AgentConfig(theta=0.3), theta=0.9)
        self.assertEqual(a._theta, 0.9)


if __name__ == "__main__":
    unittest.main()
