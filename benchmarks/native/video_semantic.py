"""Real decoded-video content and motion oracles, using the pinned CPU CLIP evaluator."""

import base64
import hashlib
import json
import statistics
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageStat

from benchmarks.native.image_semantic import ImageVerifier, cosine, rgb


def decode(raw):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as file:
        file.write(raw)
        file.flush()
        info = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_read_frames",
                    "-of",
                    "json",
                    file.name,
                ],
                timeout=60,
            )
        )["streams"][0]
        width, height, count = info["width"], info["height"], int(info["nb_read_frames"])
        if width * height * count * 3 > 512 * 1024 * 1024 or count < 2:
            raise ValueError("Unexpected decoded video dimensions or frame count")
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                file.name,
                "-map",
                "0:v:0",
                "-sws_flags",
                "bitexact+accurate_rnd",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
    stride = width * height * 3
    if len(result.stdout) != count * stride:
        raise ValueError("Decoded video frame count differs from actual container")
    frames = [
        Image.frombytes("RGB", (width, height), result.stdout[i * stride : (i + 1) * stride])
        for i in range(count)
    ]
    numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
    return frames, {
        "width": width,
        "height": height,
        "frames": count,
        "fps": numerator / denominator,
    }


def sampled_indices(count):
    return sorted({round(i * (count - 1) / 4) for i in range(5)})


def motion(frames):
    centers = []
    fractions = []
    for frame in frames:
        points = [
            (i % frame.width, i // frame.width)
            for i, (r, g, b) in enumerate(frame.getdata())
            if r > 90 and r > 1.4 * g and r > 1.4 * b
        ]
        fractions.append(len(points) / (frame.width * frame.height))
        centers.append(statistics.mean(x for x, y in points) if points else None)
    return {
        "red_fraction_by_frame": fractions,
        "red_centroid_x_by_frame": centers,
        "rightward_displacement_fraction": (centers[-1] - centers[0]) / frames[0].width
        if centers[0] is not None and centers[-1] is not None
        else -1,
        "distinct_decoded_frames": len({hashlib.sha256(f.tobytes()).hexdigest() for f in frames}),
    }


def pixel_error(first, second):
    if first.size != second.size:
        raise ValueError("Anchor dimensions differ from actual native video")
    return sum(ImageStat.Stat(ImageChops.difference(first, second)).mean) / 3 / 255


def semantic(case, output, *, assets=None):
    verification = output.get("verification")
    if verification is None:
        return {
            "status": "NOT_VERIFIED",
            "checks": {},
            "reason": "Await pinned CPU video semantic evaluation after target GPU backend shutdown",
        }
    artifact = output["artifacts"][0]
    raw = base64.b64decode(artifact["data_base64"], validate=True)
    frames, info = decode(raw)
    expected = case["expected"]
    start = expected.get("generated_segment_start_frame", 0)
    generated = frames[start:]
    if len(generated) != case["request"]["num_frames"]:
        raise ValueError("Actual generated segment length differs from requested native frames")
    indices = [start + i for i in sampled_indices(len(generated))]
    metrics = motion(generated)
    similarities = [
        [cosine(v, t) for t in verification["text_embeddings"]]
        for v in verification["frame_embeddings"]
    ]
    positives = [row[0] for row in similarities]
    margins = [row[0] - max(row[1:]) for row in similarities]
    checks = {
        "exact_real_video_identity": len(output["artifacts"]) == 1
        and output["model"] == case["model"]
        and artifact["type"] == "video"
        and artifact["mime_type"] == "video/mp4"
        and len(raw) == artifact["size_bytes"]
        and hashlib.sha256(raw).hexdigest() == artifact["sha256"] == verification["input_sha256"],
        "exact_requested_container": info
        == {
            "width": case["request"]["width"],
            "height": case["request"]["height"],
            "frames": expected.get("output_num_frames", case["request"]["num_frames"]),
            "fps": float(case["request"]["fps"]),
        },
        "exact_pinned_semantic_verifier": verification["pins_sha256"]
        == expected["verifier_pins_sha256"],
        "exact_frame_sampling": verification["frame_indices"] == indices
        and len(verification["frame_embeddings"]) == len(indices),
        "exact_text_oracle": verification["texts"]
        == [expected["positive_text"], *expected["negative_texts"]],
        "expected_object_and_scene": min(positives) >= expected["minimum_frame_text_cosine"]
        and statistics.mean(positives) >= expected["minimum_mean_text_cosine"]
        and min(margins) >= expected["minimum_text_margin"],
        "nonstatic_video": metrics["distinct_decoded_frames"]
        >= expected["minimum_distinct_frames"],
        "declared_native_seed": output["seed"] in case["seed_sequence"],
    }
    if "minimum_red_fraction" in expected:
        checks["red_object_visible_all_frames"] = (
            min(metrics["red_fraction_by_frame"]) >= expected["minimum_red_fraction"]
        )
        checks["real_rightward_motion"] = (
            metrics["rightward_displacement_fraction"]
            >= expected["minimum_rightward_displacement_fraction"]
        )
    if "minimum_mean_frame_change" in expected:
        metrics["generated_mean_frame_change"] = statistics.mean(
            pixel_error(a, b) for a, b in zip(generated, generated[1:], strict=False)
        )
        checks["generated_segment_has_real_pixel_motion"] = (
            metrics["generated_mean_frame_change"] >= expected["minimum_mean_frame_change"]
        )
    for anchor in expected.get("keyframe_anchors", []):
        reference = rgb(assets[anchor["asset"]])
        error = pixel_error(frames[anchor["frame_index"]], reference)
        metrics["anchor_mae_" + str(anchor["frame_index"])] = error
        checks["keyframe_" + str(anchor["frame_index"]) + "_preserved"] = (
            error <= anchor["maximum_mean_absolute_error"]
        )
    if "preserved_source" in expected:
        spec = expected["preserved_source"]
        source, source_info = decode(assets[spec["asset"]])
        if spec["frames"] != round(len(source) / source_info["fps"] * case["request"]["fps"]):
            raise ValueError("Declared extension source length differs from actual input")
        errors = [
            pixel_error(
                frames[round(i / source_info["fps"] * case["request"]["fps"])],
                f.resize((info["width"], info["height"]), Image.Resampling.BICUBIC),
            )
            for i, f in enumerate(source)
        ]
        metrics["source_anchor_mean_absolute_errors"] = errors
        checks["real_input_prefix_preserved"] = (
            start == spec["frames"] and max(errors) <= spec["maximum_mean_absolute_error"]
        )
    for name, minimum in expected.get("reference_cosines", {}).items():
        reference = verification["reference_embeddings"][name]
        values = [cosine(vector, reference) for vector in verification["frame_embeddings"]]
        metrics["frame_reference_cosines_" + name] = values
        checks["source_reference_identity_" + name] = (
            verification["reference_sha256"][name] == hashlib.sha256(assets[name]).hexdigest()
        )
        checks["source_reference_visible_" + name] = min(values) >= minimum
    if case["capability"] == "video.video_with_audio":
        checks["required_native_audio_semantic_contract"] = (
            "spoken_audio" in expected and "audio_verification" in output
        )
        if checks["required_native_audio_semantic_contract"]:
            from benchmarks.native.video_audio import semantic as audio_semantic

            audio = audio_semantic(
                expected["spoken_audio"],
                output["audio_verification"],
                raw,
                info["frames"] / info["fps"],
            )
            checks.update(audio["checks"])
            metrics.update(audio["metrics"])
        graph = output.get("graph", {})
        checks["joint_native_av_decode"] = graph.get("audio", {}) == {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["sample", 0], "vae": ["audio_vae", 0]},
        } and graph.get("video", {}).get("inputs", {}).get("audio") == ["audio", 0]
    if case["model"].startswith("minimax"):
        history = output["native_history"]
        messages = history["status"]["messages"]
        cached = {
            node for kind, data in messages if kind == "execution_cached" for node in data["nodes"]
        }
        checks.update(
            actual_native_execution=history["status"]["completed"] is True
            and history["status"]["status_str"] == "success"
            and {"execution_start", "execution_success"}.issubset(
                {kind for kind, data in messages}
            ),
            actual_submitted_graph=history["prompt"][2] == output["graph"]
            and history["prompt"][1] == output["prompt_id"],
            sampling_not_cached="sample" not in cached,
            exact_sampler_seed=output["graph"]["sample"]["inputs"]["seed"] == output["seed"],
        )
    metrics.update(container=info, frame_positive_cosines=positives, frame_positive_margins=margins)
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": metrics,
    }


class VideoVerifier(ImageVerifier):
    def evaluate_video(self, raw, expected):
        frames, info = decode(raw)
        start = expected.get("generated_segment_start_frame", 0)
        indices = [start + i for i in sampled_indices(len(frames) - start)]
        texts = [expected["positive_text"], *expected["negative_texts"]]
        names = list(expected.get("reference_cosines", {}))
        inputs = self.processor(
            text=texts,
            images=[frames[i] for i in indices] + [rgb(self.assets[name]) for name in names],
            return_tensors="pt",
            padding=True,
        )
        with self.torch.inference_mode():
            result = self.model(**inputs)
        return {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "pins_sha256": self.evidence["pins_sha256"],
            "frame_indices": indices,
            "texts": texts,
            "frame_embeddings": result.image_embeds[: len(indices)].cpu().tolist(),
            "reference_embeddings": {
                name: result.image_embeds[len(indices) + i].cpu().tolist()
                for i, name in enumerate(names)
            },
            "reference_sha256": {
                name: hashlib.sha256(self.assets[name]).hexdigest() for name in names
            },
            "text_embeddings": result.text_embeds.cpu().tolist(),
            "decoder_version": subprocess.check_output(
                ["ffmpeg", "-version"], text=True
            ).splitlines()[0],
        }
