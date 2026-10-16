"""Write openapi.json - feed it to openapi-generator to produce client SDKs in any language."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ntsb_api.server import app  # noqa: E402

(ROOT / "openapi.json").write_text(json.dumps(app.openapi(), indent=2))
print("wrote openapi.json with", len(app.openapi()["paths"]), "paths")
