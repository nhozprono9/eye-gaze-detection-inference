import argparse
import time
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eye gaze detection from webcam.")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO model file.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold.")
    parser.add_argument("--width", type=int, default=800, help="Output frame width.")
    parser.add_argument("--height", type=int, default=640, help="Output frame height.")
    parser.add_argument("--warning-sec", type=float, default=1.0, help="Seconds before warning text.")
    parser.add_argument("--cheating-sec", type=float, default=2.0, help="Seconds before cheating text.")
    parser.add_argument("--save", default="", help="Optional output video path (e.g. output.mp4).")
    return parser.parse_args()


def infer_direction(eye_region_bgr) -> Optional[str]:
    if eye_region_bgr.size == 0:
        return None

    gray_eye = cv2.cvtColor(eye_region_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_eye, (7, 7), 1)
    _, threshold_eye = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    cv2.circle(eye_region_bgr, (cx, cy), 5, (0, 0, 255), -1)

    eye_center_x = eye_region_bgr.shape[1] // 2
    eye_center_y = eye_region_bgr.shape[0] // 2
    if cx < eye_center_x - 10:
        return "Left"
    if cx > eye_center_x + 10:
        return "Right"
    if cy < eye_center_y - 10:
        return "Up"
    return "Center"


def clamp_box(x1: int, y1: int, x2: int, y2: int, frame_shape: Tuple[int, int, int]):
    h, w = frame_shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2


def main():
    args = parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20, (args.width, args.height))

    direction_start_time = None
    last_non_center_direction = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        resized = cv2.resize(frame, (args.width, args.height))
        flipped = cv2.flip(resized, 1)
        detections = model(flipped, verbose=False)[0]

        eye_direction = None

        for pred in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, _class_id = pred
            if score < args.conf:
                continue

            x1i, y1i, x2i, y2i = clamp_box(int(x1), int(y1), int(x2), int(y2), flipped.shape)
            if x2i <= x1i or y2i <= y1i:
                continue

            eye_region = flipped[y1i:y2i, x1i:x2i]
            direction = infer_direction(eye_region)
            if direction in {"Left", "Right", "Up"}:
                eye_direction = direction

            cv2.rectangle(flipped, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            label = f"eye {score:.2f}"
            cv2.putText(flipped, label, (x1i, max(20, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if eye_direction in {"Left", "Right", "Up"}:
            if direction_start_time is None:
                direction_start_time = time.time()
            else:
                elapsed = time.time() - direction_start_time
                if elapsed > args.cheating_sec:
                    cv2.putText(flipped, "Cheating Detected!", (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                elif elapsed > args.warning_sec:
                    cv2.putText(flipped, "Warning: Possible Cheating!", (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            last_non_center_direction = eye_direction
        else:
            if last_non_center_direction is not None:
                direction_start_time = None
                last_non_center_direction = None

        cv2.imshow("Eye Gaze Detection", flipped)
        if writer is not None:
            writer.write(flipped)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
