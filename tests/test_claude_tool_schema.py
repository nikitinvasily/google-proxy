import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import antigravity_proxy as proxy


class ClaudeToolSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {"anyOf": [
                        {"type": "object", "properties": {
                            "name": {"type": "string"},
                            "action": {"type": "string", "enum": ["create"]},
                            "content": {"type": "string"},
                        }, "required": ["name", "action", "content"],
                         "additionalProperties": False},
                        {"type": "object", "properties": {
                            "name": {"type": "string"},
                            "action": {"type": "string", "enum": ["patch"]},
                            "old_string": {"type": "string"},
                        }, "required": ["name", "action", "old_string"],
                         "additionalProperties": False},
                    ]},
                },
                "optional": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["operations"],
        }
        self.tools = [{"type": "function", "function": {
            "name": "skill_manage", "description": "Manage skills",
            "parameters": self.schema,
        }}]

    def build(self, model):
        with patch.object(proxy, "get_project_id", return_value="test-project"):
            return proxy._build_gemini_request({
                "messages": [{"role": "user", "content": "hello"}],
                "tools": self.tools,
            }, model)["request"]["tools"][0]["functionDeclarations"][0]["parameters"]

    def test_claude_flattens_nested_object_union_without_mutating_original(self):
        original = copy.deepcopy(self.schema)
        result = self.build("claude-sonnet-5-5-medium")
        item = result["properties"]["operations"]["items"]
        self.assertNotIn("anyOf", item)
        self.assertEqual(item["type"], "object")
        self.assertEqual(item["required"], ["name", "action"])
        self.assertEqual(set(item["properties"]), {"name", "action", "content", "old_string"})
        self.assertEqual(item["properties"]["action"]["enum"], ["create", "patch"])
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(self.schema, original)

    def test_claude_relaxes_nonobject_union_without_restricting_valid_inputs(self):
        result = self.build("claude-sonnet-5-5-medium")
        self.assertNotIn("anyOf", result["properties"]["optional"])
        self.assertEqual(result["properties"]["optional"], {})

    def test_object_union_accepts_array_and_object_enum_values(self):
        self.schema["properties"]["data"] = {"anyOf": [
            {"type": "object", "properties": {
                "value": {"enum": [["a"], {"key": 1}]},
            }},
            {"type": "object", "properties": {
                "value": {"enum": [["b"], {"key": 1}]},
            }},
        ]}
        result = self.build("claude-sonnet-5-5-medium")
        self.assertEqual(result["properties"]["data"]["properties"]["value"]["enum"],
                         [["a"], {"key": 1}, ["b"]])

    def test_negated_and_exclusive_unions_only_relax(self):
        self.schema["properties"]["negated"] = {
            "not": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        }
        self.schema["properties"]["exclusive"] = {
            "oneOf": [
                {"anyOf": [{"type": "string"}, {"type": "null"}]},
                {"type": "number"},
            ],
        }
        self.schema["properties"]["conditional"] = {
            "if": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "then": {"type": "string"},
        }
        result = self.build("claude-sonnet-5-5-medium")["properties"]
        self.assertEqual(result["negated"], {})
        self.assertEqual(result["exclusive"], {})
        self.assertEqual(result["conditional"], {})

    def test_literal_data_is_not_treated_as_schema(self):
        self.schema["properties"]["literal"] = {
            "type": "object",
            "default": {"anyOf": "literal client data"},
            "enum": [{"anyOf": "valid enum value"}],
        }
        result = self.build("claude-sonnet-5-5-medium")
        self.assertEqual(result["properties"]["literal"],
                         self.schema["properties"]["literal"])

    def test_gemini_keeps_original_union_schema(self):
        self.assertEqual(self.build("gemini-3-flash"), self.schema)


if __name__ == "__main__":
    unittest.main()
