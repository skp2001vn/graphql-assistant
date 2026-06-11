from __future__ import annotations

import unittest

from graphql_assistant.core.responses import PrettyJSONResponse


class ResponsesTest(unittest.TestCase):
    def test_pretty_json_response_renders_indented_json(self) -> None:
        rendered = PrettyJSONResponse(content={"operation": ["query X"], "variables": {"code": "US"}}).body

        self.assertIn(b'\n  "operation": [', rendered)
        self.assertIn(b'\n    "code": "US"', rendered)


if __name__ == "__main__":
    unittest.main()
