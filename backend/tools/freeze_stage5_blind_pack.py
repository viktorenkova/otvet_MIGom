from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from backend.tools.prepare_stage5_blind_pack import validate_reviewed_pack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path(".work/stage5-blind-review-pack.json"))
    parser.add_argument("--manifest", type=Path, default=Path("tests/data/stage5_blind_manifest.json"))
    args = parser.parse_args()
    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    readiness = validate_reviewed_pack(pack)
    if not readiness["freeze_ready"]:
        print(json.dumps({"frozen": False, "readiness": readiness}, ensure_ascii=False, indent=2))
        return 2
    raw = args.pack.read_bytes()
    dialogue_turn_count = sum(len(item.get("turns", [])) for item in pack["dialogues"])
    manifest = {
        "schema_version": 1,
        "dataset_version": pack["dataset_version"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "pack_path": str(args.pack),
        "pack_sha256": hashlib.sha256(raw).hexdigest(),
        "single_turn_count": len(pack["cases"]),
        "dialogue_count": len(pack["dialogues"]),
        "dialogue_turn_count": dialogue_turn_count,
        "review_record_count": len(pack["cases"]) + dialogue_turn_count,
        "reviewer_attestation": pack["reviewer_attestation"],
        "policy": "Labels and phrases are immutable after this manifest is committed and before the first run.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": True, "manifest": manifest}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
