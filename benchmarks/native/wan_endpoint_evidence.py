"""Additional FL2V bindings inside the receipt-verified native evidence gate."""

import re

from factory.adapters.wan.endpoint_conditioning import runtime_identity


def verify(case, sample, output, records, assets):
    matches = [
        row for row in records
        if row.get("seed") == sample["seed"]
        and row.get("output_mp4_sha256") == output["artifacts"][0]["sha256"]
    ]
    assert len(matches) == 1, "Missing unique endpoint evidence bound to actual seed/MP4"
    row = matches[0]
    assert all(row.get(key) == value for key, value in runtime_identity().items()), (
        "Experimental endpoint implementation/source differs from the selected Adapter"
    )
    request = case["request"]
    for field, label in (("image", "first"), ("last_image", "last")):
        name = request[field]
        assert name.startswith("@")
        assert row[f"source_{label}_image_sha256"] == assets[name[1:]]["sha256"], (
            "Endpoint record does not consume the actual predeclared photographed input"
        )
    frames = request["num_frames"]
    temporal = (frames - 1) // 4 + 1
    assert row["condition_shape"] == [1, 48, temporal, request["height"] // 16, request["width"] // 16]
    assert row["context_pixel_frames"] == frames
    assert row["conditioned_latent_indices"] == [0, temporal - 1], (
        "Both endpoints and only those endpoints must be fixed"
    )
    assert row["free_interior_latent_frames"] == temporal - 2 and temporal > 2
    for key in (
        "normalized_first_image_sha256", "normalized_last_image_sha256",
        "first_condition_sha256", "last_condition_sha256", "condition_sha256", "mask_sha256",
    ):
        assert re.fullmatch(r"[a-f0-9]{64}", row.get(key, "")), "Missing actual tensor identity"
    return row
