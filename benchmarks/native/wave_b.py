"""Wave B exact native ASR probe; other Wave B capabilities remain unverified."""

import argparse
import asyncio
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.request
import wave
from importlib.metadata import version

from benchmarks.native.wave_a import digest, model_run, execute_case, encoded
from factory.core.media import MediaRequest
from model_registry.deployment_validator import inspect_environment

ROOT = Path(__file__).resolve().parents[2]


def words(text):
    return re.findall(r"[a-z0-9]+", text.casefold())


def word_error_rate(reference, hypothesis):
    reference, hypothesis = words(reference), words(hypothesis)
    if not reference:
        raise ValueError("A semantic reference must contain words")
    previous = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, 1):
        current = [index]
        for j, actual in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual))
            )
        previous = current
    return previous[-1] / len(reference)


def semantic(case, output):
    if case["model"] == "qwen-vlm":
        return visual_semantic(case, output)
    if case["model"] == "qwen-tts":
        return speech_semantic(case, output)
    text = output.get("text", "")
    expected = case["expected"]
    wer = word_error_rate(expected["transcript"], text)
    normalized = " ".join(words(text))
    checks = {
        "transcript_matches_ground_truth": wer <= expected["max_word_error_rate"],
        "required_spoken_phrases": all(p in normalized for p in expected["required_phrases"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {"word_error_rate": wer, "actual_transcript": text},
    }


def speech_semantic(case, output):
    verification = output.get("verification")
    if verification is None:
        return {
            "status": "NOT_VERIFIED",
            "checks": {},
            "reason": "Await actual pinned CPU semantic verifiers after native GPU shutdown",
        }
    expected = case["expected"]
    artifact = output["artifacts"][0]
    raw = base64.b64decode(artifact["data_base64"], validate=True)
    wer = word_error_rate(expected["transcript"], verification["transcript"])
    checks = {
        "real_waveform_identity": hashlib.sha256(raw).hexdigest()
        == artifact["sha256"]
        == verification["input_sha256"]
        and len(raw) == artifact["size_bytes"],
        "native_audio_type": artifact["type"] == "audio"
        and artifact["mime_type"] == "audio/wav"
        and output["model"] == case["model"],
        "expected_spoken_words": wer <= expected["max_word_error_rate"],
        "expected_duration": expected["minimum_duration_seconds"]
        <= verification["duration_seconds"]
        <= expected["maximum_duration_seconds"],
        "exact_reference_audio": verification["reference_sha256"]
        == expected["reference_audio_sha256"],
        "exact_semantic_verifiers": verification["verifier_pins_sha256"]
        == expected["verifier_pins_sha256"],
    }
    metrics = {
        "word_error_rate": wer,
        "actual_transcript": verification["transcript"],
        "duration_seconds": verification["duration_seconds"],
    }
    if "minimum_speaker_cosine" in expected:
        ref, actual = (
            verification["speaker_reference_embedding"],
            verification["speaker_output_embedding"],
        )
        valid = len(ref) == len(actual) == 512 and all(math.isfinite(x) for x in ref + actual)
        norm_ref = math.sqrt(math.fsum(x * x for x in ref))
        norm_actual = math.sqrt(math.fsum(x * x for x in actual))
        valid = valid and norm_ref > 0 and norm_actual > 0
        similarity = (
            math.fsum(x * y for x, y in zip(ref, actual)) / (norm_ref * norm_actual)
            if valid
            else -1
        )
        checks.update(
            valid_speaker_embeddings=valid,
            reference_speaker_similarity=similarity >= expected["minimum_speaker_cosine"],
        )
        metrics["speaker_cosine"] = similarity
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
    }


def visual_semantic(case, output):
    if case["operation"] == "response":
        text = "".join(
            p.get("text", "")
            for item in output["output"]
            if item.get("type") == "message"
            for p in item.get("content", [])
            if p.get("type") == "output_text"
        )
    else:
        text = output["choices"][0]["message"].get("content") or ""
    expected = case["expected"]
    normalized = " ".join(words(text))
    metrics = {"actual_text": text}
    if "exact_text" in expected:
        checks = {"expected_answer": normalized == " ".join(words(expected["exact_text"]))}
    elif "normalized_text" in expected:
        checks = {"read_actual_document_title": normalized == expected["normalized_text"]}
    elif "bbox" in expected:
        encoded = text.strip()
        if "markdown_json_code_block" in expected.get("accepted_bbox_encodings", []):
            fenced = re.fullmatch(r"```json\s*(\{.*\})\s*```", encoded, flags=re.DOTALL)
            if fenced:
                encoded = fenced.group(1)
        box = json.loads(encoded)["bbox"]
        valid = (
            isinstance(box, list)
            and len(box) == 4
            and all(type(x) in (float, int) and 0 <= x <= 1000 for x in box)
            and box[0] < box[2]
            and box[1] < box[3]
        )
        iou = 0.0
        if valid:
            target = expected["bbox"]
            area = max(0, min(box[2], target[2]) - max(box[0], target[0])) * max(
                0, min(box[3], target[3]) - max(box[1], target[1])
            )
            union = (
                (box[2] - box[0]) * (box[3] - box[1])
                + (target[2] - target[0]) * (target[3] - target[1])
                - area
            )
            iou = area / union
        checks = {
            "valid_normalized_box": valid,
            "localizes_visible_head": iou >= expected["minimum_iou"],
        }
        metrics.update(actual_box=box, intersection_over_union=iou)
    else:
        tokens = set(words(text))
        checks = {
            "recognizes_people": bool(tokens.intersection(expected["entity_any"])),
            "recognizes_walking_action": bool(tokens.intersection(expected["action_any"])),
        }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
    }


def visual_asset(spec, directory):
    if "fixture_path" in spec:
        raw = (ROOT / spec["fixture_path"]).read_bytes()
    else:
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            raw = response.read(spec["size_bytes"] + 1)
    if len(raw) != spec["size_bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("Pinned real visual input integrity failed")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "source.bin").write_bytes(raw)
    (directory / "input_identity.json").write_text(json.dumps(spec, indent=2) + "\n")
    return "data:" + spec["mime_type"] + ";base64," + base64.b64encode(raw).decode()


def decode_asset(spec, directory):
    directory.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(spec["url"], timeout=120) as response:
        raw = response.read(spec["size_bytes"] + 1)
    if len(raw) != spec["size_bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("Pinned real-audio fixture integrity failed")
    source = directory / "source.flac"
    source.write_bytes(raw)
    pcm = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-acodec",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ],
        timeout=30,
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(spec["channels"])
        target.setsampwidth(2)
        target.setframerate(spec["sample_rate"])
        target.writeframes(pcm)
    result = buffer.getvalue()
    if (
        len(result) != spec["decoded_wav_size_bytes"]
        or hashlib.sha256(result).hexdigest() != spec["decoded_wav_sha256"]
    ):
        raise ValueError("Decoded actual WAV differs from pinned native input")
    (directory / "input_identity.json").write_text(json.dumps(spec, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    if args.models not in (["qwen-asr"], ["qwen-vlm"], ["qwen-tts"]):
        raise ValueError("Select exactly one explicit Wave B model")
    model_id = args.models[0]
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_b.json").read_text())
    cases = [c for c in suite["cases"] if c["model"] == model_id]
    assets = {c["asset"]: suite["assets"][c["asset"]] for c in cases if c.get("asset")}
    loaded_assets = {
        name: (
            decode_asset(spec, args.output / "inputs" / name)
            if model_id in {"qwen-asr", "qwen-tts"}
            else visual_asset(spec, args.output / "inputs" / name)
        )
        for name, spec in assets.items()
    }

    async def execute(context, adapter, case):
        if model_id == "qwen-vlm":
            prepared_case = copy.deepcopy(case)
            for message in prepared_case["request"].get("messages", []):
                for part in message["content"] if isinstance(message["content"], list) else []:
                    for name in ("image_url", "video_url"):
                        if name in part:
                            asset_name = part[name]["url"].removeprefix("asset:")
                            part[name]["url"] = loaded_assets[asset_name]
            return await execute_case(context, adapter, prepared_case)
        request = MediaRequest(
            model=case["model"],
            capability=case["capability"],
            inputs={
                "reference_audio" if model_id == "qwen-tts" else "audio": loaded_assets[
                    case["asset"]
                ],
                **case["request"],
            },
        )
        capability = adapter.resolve_capability(request, case["capability"])
        required = adapter.required_capabilities(request, capability) | {capability}
        if any(
            not context.model.capabilities.get(cid) or cid not in adapter.capabilities
            for cid in required
        ):
            raise ValueError("Undeclared speech capability")
        adapter.validate_request(context.model, request, capability)
        prepared = await asyncio.to_thread(adapter.prepare, context, request, capability)
        raw = await asyncio.to_thread(adapter.infer, context, prepared)
        normalized = await asyncio.to_thread(adapter.normalize_output, context, prepared, raw)
        if model_id == "qwen-tts":
            for item in normalized["artifacts"]:
                item["data_base64"] = base64.b64encode(item.pop("data")).decode()
        return normalized

    report = {
        "classification": "REAL_NATIVE_VALIDATION_ATTEMPT",
        "hardware": inspect_environment().gpus,
        "cases_sha256": digest({"cases": cases, "assets": assets}),
        "case_definitions": cases,
        "input_assets": assets,
        "runtime_versions": {
            name: version(name)
            for name in (
                ["qwen-asr"]
                if model_id == "qwen-asr"
                else ["vllm", "vllm-omni"]
                if model_id == "qwen-tts"
                else ["vllm"]
            )
            + ["torch", "transformers", "huggingface-hub", "psutil"]
        },
        "models": [],
        "fallback": False,
        "production_ready": False,
        "full_factory_e2e_executed": False,
    }
    assert set(report["hardware"]) == {0, 1} and all(
        "t4" in v["name"].lower() for v in report["hardware"].values()
    )
    directory = args.output / model_id
    directory.mkdir(exist_ok=True)
    result = asyncio.run(
        model_run(model_id, args.root, directory, cases, executor=execute, validator=semantic)
    )
    if model_id == "qwen-tts" and any(c["runs"] for c in result["cases"]):
        try:
            from benchmarks.native.speech_semantic import SpeechVerifier

            verifier = SpeechVerifier(args.root / "semantic_weights", loaded_assets["jfk"])
            report["semantic_verifiers"] = verifier.evidence
            for native_case in result["cases"]:
                case = next(c for c in cases if c["id"] == native_case["case_id"])
                for run in native_case["runs"]:
                    path = directory / run["output_file"]
                    actual = json.loads(path.read_text())
                    raw = base64.b64decode(actual["artifacts"][0]["data_base64"], validate=True)
                    actual["verification"] = verifier.evaluate(raw)
                    path.write_bytes(encoded(actual))
                    run["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    run["semantic_validation"] = semantic(case, actual)
                native_case["status"] = (
                    "PASS"
                    if len(native_case["runs"]) == 4
                    and all(
                        r["semantic_validation"]["status"] == "PASS" for r in native_case["runs"]
                    )
                    else "NOT_MET"
                )
            result["status"] = (
                "PASS"
                if result.get("stop_verified")
                and len(result["cases"]) == len(cases)
                and all(c["status"] == "PASS" for c in result["cases"])
                else "NOT_MET"
            )
        except Exception as exc:
            import traceback

            result["semantic_verifier_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            result["status"] = "NOT_MET"
        (directory / "observation.json").write_text(json.dumps(result, indent=2) + "\n")
    report["models"].append(result)
    report["status"] = result["status"]
    (args.output / "wave_b.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": report["status"], "native_scope": model_id + "_ONLY_NOT_FULL_WAVE_B"})
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
