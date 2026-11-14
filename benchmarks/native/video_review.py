"""Export every collected native frame for direct review; never grants acceptance."""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw

from benchmarks.native.video_semantic import decode

ROOT = Path(__file__).resolve().parents[2]


def export(kind, version, output=None):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", kind) or version < 1:
        raise ValueError("Explicit native kind and positive version required")
    folder = ROOT / f"validation/native_{kind}_v{version}"
    receipt = json.loads((ROOT / f"validation/native_{kind}_receipt_v{version}.json").read_text())
    hashes = {}
    for entry in receipt["artifacts"]:
        path = (ROOT / entry["file"]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Receipt artifact is outside this exact native run")
        hashes[path] = entry["sha256"]

    def verified(path):
        path = path.resolve()
        if path not in hashes:
            raise ValueError("Native artifact is not listed in this collected receipt")
        raw = path.read_bytes()
        if hashes.get(path) != hashlib.sha256(raw).hexdigest():
            raise ValueError("Collected native artifact digest mismatch")
        return raw

    run_path = ROOT / f"validation/native_{kind}_run_v{version}.json"
    if not run_path.is_file():
        run_path = ROOT / f"validation/native_{kind}_run.json"
    run_raw = run_path.read_bytes()
    run = json.loads(run_raw)
    if (
        run["response"]["versionNumber"] != version
        or hashlib.sha256(run_raw).hexdigest() != receipt["run_record_sha256"]
    ):
        raise ValueError("Receipt does not match the exact submitted version")
    observed = json.loads(verified(folder / "wave_d.json"))
    destination = (
        Path(output) if output else ROOT / f"validation/native_video_reviews/{kind}_v{version}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "classification": "DERIVED_ALL_FRAME_CONTACT_SHEETS_ONLY_NOT_NEW_INFERENCE_OR_ACCEPTANCE",
        "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_ready": False,
        "cases": [],
    }
    for model in observed["models"]:
        for case in model["cases"]:
            case_id = case["case_id"]
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", case_id):
                raise ValueError("Unsafe native case identity")
            row = {"run_id": f"kaggle:{run['slug']}:v{version}:{case_id}", "outputs": []}
            case_dir = destination / case_id
            case_dir.mkdir(exist_ok=True)
            for sample in case["runs"]:
                filename = sample["output_file"]
                if Path(filename).name != filename:
                    raise ValueError("Native output must be a direct model artifact")
                raw = verified(folder / model["model"] / filename)
                digest = hashlib.sha256(raw).hexdigest()
                if digest != sample["output_sha256"]:
                    raise ValueError("Case output digest differs from receipt")
                result = json.loads(raw)
                if len(result["artifacts"]) != 1 or result["model"] != model["model"]:
                    raise ValueError("Unexpected native output identity")
                artifact = result["artifacts"][0]
                video = base64.b64decode(artifact["data_base64"], validate=True)
                if (
                    artifact["type"] != "video"
                    or artifact["mime_type"] != "video/mp4"
                    or len(video) != artifact["size_bytes"]
                    or hashlib.sha256(video).hexdigest() != artifact["sha256"]
                ):
                    raise ValueError("Native MP4 bytes differ from recorded identity")
                frames, info = decode(video)
                width, height, columns = 192, 128, 8
                sheet = Image.new(
                    "RGB",
                    (columns * width, ((len(frames) + columns - 1) // columns) * (height + 18)),
                    "white",
                )
                draw = ImageDraw.Draw(sheet)
                for index, frame in enumerate(frames):
                    thumbnail = frame.copy()
                    thumbnail.thumbnail((width, height), Image.Resampling.LANCZOS)
                    x, y = index % columns * width, index // columns * (height + 18)
                    sheet.paste(thumbnail, (x, y))
                    draw.text((x + 2, y + height), str(index), fill="black")
                target = case_dir / f"all_frames_{sample['iteration']}.png"
                sheet.save(target)
                row["outputs"].append(
                    {
                        "iteration": sample["iteration"],
                        "role": sample["role"],
                        "source_output_json_sha256": digest,
                        "native_mp4_sha256": artifact["sha256"],
                        "decoded": info,
                        "all_frame_indices": list(range(len(frames))),
                        "contact_sheet": str(target.resolve()),
                        "contact_sheet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    }
                )
            manifest["cases"].append(row)
    (destination / "contact_sheets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = export(args.kind, args.version, args.output)
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "cases": [
                    {
                        "run_id": case["run_id"],
                        "review_sheets": [o["contact_sheet"] for o in case["outputs"]],
                    }
                    for case in report["cases"]
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
