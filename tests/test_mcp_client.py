import unittest

from ia_claude.mcp.mcp_client import _remove_schema_dialect


class McpClientTests(unittest.TestCase):
    def test_remove_schema_dialect_recursively_without_mutating_source(self):
        source = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "item": {
                    "$schema": "nested dialect",
                    "type": "string",
                }
            },
        }

        cleaned = _remove_schema_dialect(source)

        self.assertNotIn("$schema", cleaned)
        self.assertNotIn("$schema", cleaned["properties"]["item"])
        self.assertIn("$schema", source)
        self.assertEqual(cleaned["properties"]["item"]["type"], "string")


if __name__ == "__main__":
    unittest.main()
