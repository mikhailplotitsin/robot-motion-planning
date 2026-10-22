import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 1280
HEIGHT = 720
FPS = 6
WARMUP_FRAMES = 30

DATA_DIR = Path(__file__).resolve().parent / "data"
WINDOW_NAME = "RealSense D415 - RGB capture"


def unique_run_directory(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = root / timestamp
    number = 1
    while candidate.exists():
        candidate = root / f"{timestamp}_{number:02d}"
        number += 1
    candidate.mkdir()
    return candidate


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    started = False
    run_dir = None
    metadata = {}

    try:
        print("Connecting to RealSense D415...")
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"Connected: {device_name}, serial {serial_number}")
        print("Only RGB stream is used (no depth).")

        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        run_dir = unique_run_directory(DATA_DIR)
        metadata = {
            "run_id": run_dir.name,
            "status": "in_progress",
            "camera": device_name,
            "serial_number": serial_number,
            "rgb_width": WIDTH,
            "rgb_height": HEIGHT,
            "rgb_fps": FPS,
            "rgb_format": "BGR8",
            "marker_dictionary": "DICT_4X4_50",
            "marker_id": 0,
            "marker_size_mm": None,  # Set to actually printed and measured size in measure.py
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "frames": {},
        }
        write_json(run_dir / "metadata.json", metadata)
        print(f"Run folder: {run_dir}")
        print("Focus the preview window. Press ENTER to capture BEFORE, then AFTER.")
        print("Press ESC or Q to cancel.")

        step = 0
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 540)

        while step < 2:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            preview = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
            message = "ENTER: capture BEFORE" if step == 0 else "Move object, then ENTER: capture AFTER"
            cv2.rectangle(preview, (0, 0), (960, 48), (0, 0, 0), -1)
            cv2.putText(preview, message, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.76, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                print("Capture cancelled. Incomplete run is marked in metadata.json.")
                metadata["status"] = "cancelled"
                write_json(run_dir / "metadata.json", metadata)
                return
            if key not in (10, 13):
                continue

            label = "before" if step == 0 else "after"
            path = run_dir / f"{label}.png"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Failed to save image: {path}")

            metadata["frames"][label] = {
                "filename": path.name,
                "captured_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                "camera_timestamp_ms": color_frame.get_timestamp(),
            }
            write_json(run_dir / "metadata.json", metadata)
            print(f"Saved {path.name}")
            step += 1

        metadata["status"] = "complete"
        metadata["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        write_json(run_dir / "metadata.json", metadata)
        print(f"Complete. Two original images and metadata saved to:\n{run_dir}")

    except Exception as exc:
        print(f"Capture error: {exc}")
        print("Check USB connection, camera availability, and requested RGB mode.")
        if run_dir is not None and metadata:
            metadata["status"] = "error"
            metadata["error"] = str(exc)
            write_json(run_dir / "metadata.json", metadata)
        raise
    finally:
        cv2.destroyAllWindows()
        if started:
            pipeline.stop()


if __name__ == "__main__":
    main()