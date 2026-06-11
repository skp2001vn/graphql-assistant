from __future__ import annotations

import json
from typing import Any

from fastapi.responses import JSONResponse


class PrettyJSONResponse(JSONResponse):
    """JSON response that renders readable indented output."""

    def render(self, content: Any) -> bytes:
        """Serialize response content with stable pretty formatting."""
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ).encode("utf-8")
