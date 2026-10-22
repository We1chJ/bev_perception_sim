"""Export Waymo segments for the web viewer in viewer/.

For each segment, writes resized camera frames and a meta.json with per-frame
3D labels and metrics (ego speed, object counts, nearest object distance).

    python tools/export_viewer.py data/waymo/raw/*.tfrecord
    python -m http.server 8765 --directory viewer
"""
import argparse
import io
import json
import math
import struct
from pathlib import Path

from PIL import Image

from inspect_segment import CAMERAS, TYPES, decode_frame, fields, records, scene_stats

TYPE_ORDER = ["VEHICLE", "PEDESTRIAN", "CYCLIST", "SIGN"]


def ego_position(frame):
    """Global (x, y, z) of the vehicle from Frame.pose, a row-major 4x4 transform."""
    for number, pose in fields(frame):
        if number == 3:  # Frame.pose
            values = []
            for t_number, t_value in fields(pose):
                if t_number == 1:  # Transform.transform (packed or unpacked doubles)
                    if isinstance(t_value, float):
                        values.append(t_value)
                    else:
                        values.extend(struct.unpack(f"<{len(t_value) // 8}d", t_value))
            return values[3], values[7], values[11]
    return None


def export_segment(path, out_root, width, quality):
    key = path.stem.replace("segment-", "").split("_with_camera_labels")[0]
    seg_dir = out_root / key
    frames_raw = list(records(path))
    stats = scene_stats(frames_raw[0])

    frames, track_ids = [], {name: set() for name in TYPE_ORDER}
    for index, raw in enumerate(frames_raw):
        timestamp, images, labels = decode_frame(raw)
        for camera, jpeg in images.items():
            image = Image.open(io.BytesIO(jpeg))
            image.draft("RGB", (width, width))
            image = image.convert("RGB")
            image = image.resize((width, round(image.height * width / image.width)), Image.BILINEAR)
            cam_dir = seg_dir / "cam" / camera
            cam_dir.mkdir(parents=True, exist_ok=True)
            image.save(cam_dir / f"{index:03d}.jpg", quality=quality)

        counts = {name: 0 for name in TYPE_ORDER}
        nearest = {"VEHICLE": None, "PEDESTRIAN": None}
        boxes = []
        for label in labels:
            name = label["type"]
            if name not in counts:
                continue
            counts[name] += 1
            track_ids[name].add(label["id"])
            distance = math.hypot(label["x"], label["y"])
            if name in nearest and (nearest[name] is None or distance < nearest[name]):
                nearest[name] = distance
            boxes.append([TYPE_ORDER.index(name), round(label["x"], 2), round(label["y"], 2),
                          round(label["length"], 2), round(label["width"], 2), round(label["heading"], 3)])
        frames.append({"timestamp": timestamp, "position": ego_position(raw), "counts": counts,
                       "nearest": {k: (round(v, 2) if v is not None else None) for k, v in nearest.items()},
                       "boxes": boxes})
        print(f"\r{key}: frame {index + 1}/{len(frames_raw)}", end="", flush=True)
    print()

    start = frames[0]["timestamp"]
    for i, frame in enumerate(frames):
        a, b = (frames[i - 1], frame) if i > 0 else (frame, frames[1])
        dt = (b["timestamp"] - a["timestamp"]) / 1e6
        frame["t"] = round((frame["timestamp"] - start) / 1e6, 3)
        frame["speed"] = round(math.dist(a["position"][:2], b["position"][:2]) / dt, 3)  # m/s
    for frame in frames:
        del frame["timestamp"], frame["position"]

    speeds = [f["speed"] for f in frames]
    meta = {
        "key": key,
        "file": path.name,
        "cameras": list(CAMERAS.values()),
        "types": TYPE_ORDER,
        "summary": {
            **stats,
            "frames": len(frames),
            "duration_s": frames[-1]["t"],
            "tracked_objects": {name: len(ids) for name, ids in track_ids.items()},
            "mean_speed_mps": round(sum(speeds) / len(speeds), 2),
            "max_speed_mps": round(max(speeds), 2),
        },
        "frames": frames,
    }
    (seg_dir / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    return {"key": key, "location": stats.get("location", ""), "frames": len(frames)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segments", type=Path, nargs="+", help="segment .tfrecord files")
    parser.add_argument("--out", type=Path, default=Path("viewer/data"), help="output directory (default: viewer/data)")
    parser.add_argument("--width", type=int, default=640, help="camera frame width in pixels (default: 640)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (default: 80)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    index = [export_segment(path, args.out, args.width, args.quality) for path in args.segments]
    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {len(index)} segment(s) to {args.out}/")


if __name__ == "__main__":
    main()
