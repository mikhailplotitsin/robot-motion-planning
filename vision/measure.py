"""
Examples, run from repo root:
    python vision/measure.py --latest --marker-mm 40 --dx 100 --dy 50
    python vision/measure.py --run vision/data/2026-10-22_14-30-00 --marker-mm 42 --dx 100 --dy 0

IMPORTANT: This is an APPROXIMATE planar measurement. Assumes a fixed camera,
near-frontal marker, constant marker height, negligible lens distortion, and
that target dx/dy are expressed in IMAGE axes: +X right, +Y down. It does NOT
calibrate the camera to the robot frame or provide a precision guarantee.
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

VISION_DIR = Path(__file__).resolve().parent
DATA_DIR = VISION_DIR / "data"
MARKER_ID = 0
DICTIONARY_NAME = "DICT_4X4_50"
CSV_COLUMNS = [
    "test_id", "status", "marker_id", "marker_mm", "dx_target_mm", "dy_target_mm",
    "dx_measured_mm", "dy_measured_mm", "error_x_mm", "error_y_mm",
    "error_norm_mm", "scale_before_mm_px", "scale_after_mm_px",
    "scale_difference_percent", "processed_at",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Measure ArUco displacement from two RGB pictures.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--latest", action="store_true", help="Use latest complete capture in vision/data")
    source.add_argument("--run", type=Path, help="Path to capture directory")
    parser.add_argument("--marker-mm", type=float, required=True,
                        help="Physical OUTER black-border marker size, in mm (e.g. 40 or 42)")
    parser.add_argument("--dx", type=float, required=True, help="Target X displacement in mm (+right in image)")
    parser.add_argument("--dy", type=float, required=True, help="Target Y displacement in mm (+down in image)")
    parser.add_argument("--scale-tolerance", type=float, default=5.0,
                        help="Warn if marker scales differ by more than this percentage (default: 5)")
    args = parser.parse_args()
    if not (0 < args.marker_mm < 10000):
        parser.error("--marker-mm must be positive")
    if args.scale_tolerance <= 0:
        parser.error("--scale-tolerance must be positive")
    return args


def get_complete_run(args):
    if args.run is not None:
        run = args.run.expanduser().resolve()
        if not run.is_dir():
            raise FileNotFoundError(f"Capture directory not found: {run}")
        return run
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"No data directory: {DATA_DIR}")
    candidates = []
    for run in DATA_DIR.iterdir():
        if not run.is_dir():
            continue
        if not (run / "before.png").is_file() or not (run / "after.png").is_file():
            continue
        metadata_file = run / "metadata.json"
        if metadata_file.is_file():
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if metadata.get("status") != "complete":
                continue
        candidates.append(run)
    if not candidates:
        raise FileNotFoundError("No completed capture pairs found in vision/data")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def detect_marker(image, marker_id):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or marker_id not in ids.flatten():
        raise ValueError(f"ArUco ID {marker_id} was not detected")
    found = ids.flatten().tolist()
    if found.count(marker_id) != 1:
        raise ValueError(f"ArUco ID {marker_id} detected multiple times")
    index = found.index(marker_id)
    points = corners[index].reshape(4, 2).astype(np.float64)
    edges = [np.linalg.norm(points[(i + 1) % 4] - points[i]) for i in range(4)]
    side_px = float(np.mean(edges))
    if side_px < 10:
        raise ValueError("Marker is too small in the image for a reliable measurement")
    center = points.mean(axis=0)
    return {"corners": points, "center_px": center, "side_px": side_px}


def label_image(image, detected, stage, k, marker_mm):
    output = image.copy()
    pts = np.rint(detected["corners"]).astype(np.int32)
    cv2.polylines(output, [pts], True, (0, 220, 0), 3, cv2.LINE_AA)
    cx, cy = (float(v) for v in detected["center_px"])
    point = (int(round(cx)), int(round(cy)))
    cv2.drawMarker(output, point, (0, 0, 255), cv2.MARKER_CROSS, 25, 3, cv2.LINE_AA)
    lines = [
        f"{stage} | ArUco {DICTIONARY_NAME} / ID {MARKER_ID}",
        f"Center: u={cx:.2f} px, v={cy:.2f} px",
        f"Marker: {marker_mm:g} mm | scale: {k:.5f} mm/px",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(output, (10, 10), (min(image.shape[1]-10, 1120), 120), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(output, line, (22, 39 + i * 32), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def update_summary(csv_path, record):
    # Replace record for the same test_id, so re-processing doesn't create duplicates.
    rows = []
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    rows = [row for row in rows if row.get("test_id") != record["test_id"]]
    rows.append(record)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    run = get_complete_run(args)
    before = cv2.imread(str(run / "before.png"), cv2.IMREAD_COLOR)
    after = cv2.imread(str(run / "after.png"), cv2.IMREAD_COLOR)
    if before is None or after is None:
        raise ValueError("Could not read before.png or after.png")
    if before.shape != after.shape:
        raise ValueError("Before and after images have different resolutions")

    first = detect_marker(before, MARKER_ID)
    second = detect_marker(after, MARKER_ID)
    k1 = args.marker_mm / first["side_px"]
    k2 = args.marker_mm / second["side_px"]
    discrepancy_pct = 100 * abs(k1 - k2) / ((k1 + k2) / 2)
    k = (k1 + k2) / 2

    # Planar approximation only. Origin is same image coordinate system for both images.
    displacement_px = second["center_px"] - first["center_px"]
    measured = displacement_px * k
    target = np.array([args.dx, args.dy], dtype=np.float64)
    error = measured - target
    error_norm = float(np.linalg.norm(error))
    flags = []
    if discrepancy_pct > args.scale_tolerance:
        flags.append("marker scale differs between images: verify constant height/tilt")

    annotated_before = label_image(before, first, "BEFORE", k1, args.marker_mm)
    annotated_after = label_image(after, second, "AFTER", k2, args.marker_mm)
    comparison = annotated_after.copy()
    p1 = tuple(np.rint(first["center_px"]).astype(int))
    p2 = tuple(np.rint(second["center_px"]).astype(int))
    cv2.arrowedLine(comparison, p1, p2, (255, 80, 0), 3, cv2.LINE_AA, tipLength=0.035)
    cv2.drawMarker(comparison, p1, (255, 80, 0), cv2.MARKER_CROSS, 25, 3, cv2.LINE_AA)
    cv2.putText(comparison, f"dx={measured[0]:+.2f}, dy={measured[1]:+.2f}, error={error_norm:.2f} mm",
                (22, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    for filename, img in [
        ("before_annotated.png", annotated_before),
        ("after_annotated.png", annotated_after),
        ("comparison.png", comparison),
    ]:
        if not cv2.imwrite(str(run / filename), img):
            raise OSError(f"Failed to save {filename}")

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    result = {
        "test_id": run.name,
        "status": "warning" if flags else "ok",
        "measurement_method": "approximate_single_marker_planar_scale",
        "coordinate_convention": "+X to the right in image, +Y downward; not robot coordinates",
        "marker_dictionary": DICTIONARY_NAME,
        "marker_id": MARKER_ID,
        "marker_size_mm": args.marker_mm,
        "center_before_px": first["center_px"].tolist(),
        "center_after_px": second["center_px"].tolist(),
        "marker_side_before_px": first["side_px"],
        "marker_side_after_px": second["side_px"],
        "scale_before_mm_px": k1,
        "scale_after_mm_px": k2,
        "scale_used_mm_px": k,
        "scale_difference_percent": discrepancy_pct,
        "target_displacement_mm": target.tolist(),
        "measured_displacement_mm": measured.tolist(),
        "error_xy_mm": error.tolist(),
        "error_norm_mm": error_norm,
        "warnings": flags,
        "processed_at": now,
    }
    write_json(run / "result.json", result)

    summary = {
        "test_id": run.name, "status": result["status"],
        "marker_id": MARKER_ID, "marker_mm": args.marker_mm,
        "dx_target_mm": args.dx, "dy_target_mm": args.dy,
        "dx_measured_mm": round(float(measured[0]), 4),
        "dy_measured_mm": round(float(measured[1]), 4),
        "error_x_mm": round(float(error[0]), 4),
        "error_y_mm": round(float(error[1]), 4),
        "error_norm_mm": round(error_norm, 4),
        "scale_before_mm_px": round(k1, 8),
        "scale_after_mm_px": round(k2, 8),
        "scale_difference_percent": round(discrepancy_pct, 3),
        "processed_at": now,
    }
    update_summary(DATA_DIR / "summary.csv", summary)

    print(f"\nRun: {run.name}")
    print(f"Marker: {DICTIONARY_NAME}, ID {MARKER_ID}, physical size {args.marker_mm:g} mm")
    print(f"Before: ({first['center_px'][0]:.2f}, {first['center_px'][1]:.2f}) px")
    print(f"After:  ({second['center_px'][0]:.2f}, {second['center_px'][1]:.2f}) px")
    print(f"Scales: {k1:.6f} and {k2:.6f} mm/px (difference {discrepancy_pct:.2f}%)")
    print(f"Target:   dx={args.dx:+.3f}, dy={args.dy:+.3f} mm")
    print(f"Measured: dx={measured[0]:+.3f}, dy={measured[1]:+.3f} mm")
    print(f"Error:    ex={error[0]:+.3f}, ey={error[1]:+.3f}, norm={error_norm:.3f} mm")
    if flags:
        for warning in flags:
            print("WARNING:", warning)
    print(f"Saved annotated images and result.json to {run}")
    print(f"Updated {DATA_DIR / 'summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"Measurement error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
