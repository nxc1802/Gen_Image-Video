"""Bind direct visual video reviews to immutable collected outputs; never inference evidence."""

import json
from pathlib import Path

CLASSIFICATION = "DIRECT_VISUAL_REVIEW_OF_ALL_DECODED_NATIVE_FRAMES_NOT_NEW_INFERENCE"


def apply(result, root):
    path = Path(root) / "validation/native_video_content_review.json"
    report = json.loads(path.read_text()) if path.is_file() else {}
    reviews = report.get("cases", {}) if report.get("classification") == CLASSIFICATION else {}
    for case in result["cases"]:
        if case["status"] != "PASS":
            continue
        review = reviews.get(case["run_id"], {})
        matched = (
            review.get("output_json_sha256") == case["output_sha256"]
            and len(case["output_sha256"]) == 4
        )
        passed = (
            matched
            and review.get("all_decoded_frames_reviewed") is True
            and review.get("decision") == "PASS"
            and bool(review.get("observation", "").strip())
        )
        case["checks"].append(
            {"check": "direct_visual_content_review_all_four_outputs", "passed": passed}
        )
        case["visual_content_review"] = review if matched else {"decision": "NOT_REVIEWED"}
        if not passed:
            case.update(status="NOT_MET", native_inference_verified=False)
            result["findings"].append(
                case["run_id"] + ": direct visual content review missing or failed"
            )
            result["exit_gate"] = "NOT_MET"
    return result
