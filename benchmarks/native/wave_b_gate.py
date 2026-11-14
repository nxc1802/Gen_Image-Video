"""Native Wave B progress. Partial ASR evidence cannot complete the VLM/Speech wave."""

import json
from pathlib import Path

from benchmarks.native.wave_a_gate import evaluate
from benchmarks.native.wave_b import semantic

ROOT = Path(__file__).resolve().parents[2]


def main():
    result = evaluate(
        wave="b",
        models=("qwen-vlm", "qwen-asr", "qwen-tts"),
        kinds=("asr", "vlm", "tts"),
        validator=semantic,
    )
    previous = json.loads((ROOT / "validation/native_wave_a_gate.json").read_text())
    if previous["exit_gate"] != "PASS":
        result["exit_gate"] = "NOT_MET"
        result["findings"].append("Wave A dependency gate is not PASS")
    (ROOT / "validation/native_wave_b_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "exit_gate": result["exit_gate"],
                "native_pass": sum(c["status"] == "PASS" for c in result["cases"]),
                "findings": result["findings"],
            }
        )
    )
    return 0 if result["exit_gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
