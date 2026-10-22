"""Inspect one frame of a Waymo Open Dataset v1.4.x perception segment.

Saves the five camera JPEGs and a bird's-eye-view (BEV) plot of the 3D labels,
and prints a summary of the segment and frame. Reads the TFRecord and protobuf
wire format directly, so the Waymo package and TensorFlow are not needed.

    python tools/inspect_segment.py data/waymo/raw/<segment>.tfrecord --frame 50
"""
import argparse
import math
import mmap
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAMERAS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
TYPES = {0: "UNKNOWN", 1: "VEHICLE", 2: "PEDESTRIAN", 3: "SIGN", 4: "CYCLIST"}
COLORS = {"VEHICLE": "tab:blue", "PEDESTRIAN": "tab:red", "CYCLIST": "tab:green",
          "SIGN": "tab:orange", "UNKNOWN": "tab:gray"}


# --- protobuf wire format ---------------------------------------------------

def _varint(b, i):
    result = shift = 0
    while True:
        c = b[i]
        i += 1
        result |= (c & 0x7F) << shift
        if c < 0x80:
            return result, i
        shift += 7


def fields(b):
    """Yield (field_number, value) pairs of one protobuf message."""
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, i = _varint(b, i)
        elif wire_type == 1:
            value, i = struct.unpack_from("<d", b, i)[0], i + 8
        elif wire_type == 2:
            length, i = _varint(b, i)
            value, i = b[i:i + length], i + length
        elif wire_type == 5:
            value, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire_type}")
        yield number, value


def records(path):
    """Yield the raw bytes of each record in a TFRecord file."""
    with open(path, "rb") as fh:
        data = memoryview(mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))
    i = 0
    while i < len(data):
        (length,) = struct.unpack_from("<Q", data, i)
        yield data[i + 12:i + 12 + length]
        i += 12 + length + 4


# --- Waymo Frame decoding ---------------------------------------------------

def scene_stats(frame):
    for number, context in fields(frame):
        if number == 1:  # Frame.context
            for c_number, stats in fields(context):
                if c_number == 4:  # Context.stats
                    names = {2: "time_of_day", 3: "location", 4: "weather"}
                    return {names[n]: bytes(v).decode() for n, v in fields(stats) if n in names}
    return {}


def decode_frame(frame):
    images, labels, timestamp = {}, [], None
    for number, value in fields(frame):
        if number == 2:  # Frame.timestamp_micros
            timestamp = value
        elif number == 4:  # Frame.images (CameraImage)
            parts = dict(fields(value))
            images[CAMERAS.get(parts[1], str(parts[1]))] = bytes(parts[2])
        elif number == 6:  # Frame.laser_labels (Label)
            label = {"type": "UNKNOWN", "id": ""}
            for l_number, l_value in fields(value):
                if l_number == 1:  # Label.box
                    box_names = {1: "x", 2: "y", 3: "z", 4: "width", 5: "length", 6: "height", 7: "heading"}
                    label.update({box_names[n]: v for n, v in fields(l_value) if n in box_names})
                elif l_number == 3:
                    label["type"] = TYPES.get(l_value, str(l_value))
                elif l_number == 4:
                    label["id"] = bytes(l_value).decode()
            labels.append(label)
    return timestamp, images, labels


# --- output -----------------------------------------------------------------

def box_corners(label):
    """Footprint corners of a box in the vehicle frame (x forward, y left)."""
    c, s = math.cos(label["heading"]), math.sin(label["heading"])
    half_l, half_w = label["length"] / 2, label["width"] / 2
    return [(label["x"] + dx * c - dy * s, label["y"] + dx * s + dy * c)
            for dx, dy in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]


def plot_bev(labels, title, out_path, extent):
    """Top-down view with forward pointing up and left pointing left."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for label in labels:
        corners = box_corners(label)
        color = COLORS.get(label["type"], "tab:gray")
        ax.fill([-y for _, y in corners], [x for x, _ in corners],
                facecolor=color, edgecolor=color, alpha=0.35, linewidth=1.2)
        front_x, front_y = (corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2
        ax.plot([-label["y"], -front_y], [label["x"], front_x], color=color, linewidth=1)
    ax.fill([-1, 1, 1, -1], [2.4, 2.4, -2.4, -2.4], color="black")  # ego vehicle
    for name, color in COLORS.items():
        if any(label["type"] == name for label in labels):
            ax.plot([], [], "s", color=color, label=name)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xlabel="left  <-  lateral (m)  ->  right", ylabel="forward (m)", title=title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("segment", type=Path, help="path to a segment .tfrecord")
    parser.add_argument("--frame", type=int, default=0, help="frame index to inspect (default: 0)")
    parser.add_argument("--out", type=Path, default=Path("out"), help="output directory (default: out/)")
    parser.add_argument("--extent", type=float, default=60.0, help="BEV half-width in meters (default: 60)")
    args = parser.parse_args()

    frames = list(records(args.segment))
    if not 0 <= args.frame < len(frames):
        parser.error(f"--frame must be between 0 and {len(frames) - 1}")
    timestamp, images, labels = decode_frame(frames[args.frame])
    start, _, _ = decode_frame(frames[0])

    out_dir = args.out / args.segment.stem / f"frame_{args.frame:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for camera, jpeg in images.items():
        (out_dir / f"{camera}.jpg").write_bytes(jpeg)
    title = f"{args.segment.stem[:28]}...  frame {args.frame}  t={(timestamp - start) / 1e6:.1f}s"
    plot_bev(labels, title, out_dir / "bev.png", args.extent)

    print(f"segment: {args.segment.name}")
    print(f"frames:  {len(frames)}   scene: {scene_stats(frames[0])}")
    print(f"frame {args.frame}: t={(timestamp - start) / 1e6:.1f}s, {len(images)} cameras, {len(labels)} 3D labels")
    counts = {}
    for label in labels:
        counts[label["type"]] = counts.get(label["type"], 0) + 1
    for name in sorted(counts):
        print(f"  {name:<11}{counts[name]}")
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
