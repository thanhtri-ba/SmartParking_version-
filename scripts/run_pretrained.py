"""Giai doan 3: chay YOLO pretrained tren anh/video, do thoi gian va VRAM.

Vi du (chay tu E:\\SmartParkingAI):
  python scripts\\run_pretrained.py                                   # anh/video trong datasets\\samples (rong -> anh mau)
  python scripts\\run_pretrained.py --source datasets\\samples\\cong.mp4 --track
  python scripts\\run_pretrained.py --source 0 --track --max-frames 300 --show   # webcam
"""
import os

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # khong cho ultralytics tu pip install (tranh keo torch CPU)

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
DEFAULT_CLASSES = "0,2,3,5,7"  # COCO: person, car, motorcycle, bus, truck
WARMUP_FRAMES = 5  # bo qua khi tinh thong ke video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(ROOT / "datasets" / "samples"), help="file, thu muc, hoac so webcam (0)")
    p.add_argument("--model", default=str(ROOT / "models" / "yolo11n.pt"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--classes", default=DEFAULT_CLASSES, help="'0,2,3,5,7' hoac 'all'")
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true", help="FP16 khi infer")
    p.add_argument("--track", action="store_true", help="tracking cho video (can package 'lap')")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--max-frames", type=int, default=0, help="0 = het video")
    p.add_argument("--show", action="store_true", help="hien cua so, nhan q de dung")
    return p.parse_args()


def collect_sources(source):
    if source.isdigit():
        return [], [int(source)], False
    path = Path(source)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.is_file())
    elif path.is_file():
        files = [path]
    else:
        sys.exit(f"LOI: khong tim thay source: {path}")
    images = [f for f in files if f.suffix.lower() in IMG_EXT]
    videos = [f for f in files if f.suffix.lower() in VID_EXT]
    if not images and not videos:
        print(f"[info] {path} chua co anh/video -> dung anh mau cua Ultralytics (bus.jpg, zidane.jpg)")
        images = sorted(ASSETS.glob("*.jpg"))
        return images, [], True
    return images, videos, False


def stats(values):
    if not values:
        return {}
    v = sorted(values)
    return {
        "mean": round(sum(v) / len(v), 1),
        "p50": round(v[len(v) // 2], 1),
        "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))], 1),
        "max": round(v[-1], 1),
    }


def run_images(model, files, kw, out_dir):
    rows = []
    for f in files:
        model.predict(str(f), **kw)  # warm-up rieng tung anh: shape moi -> cuDNN chon lai kernel, lan dau rat cham
        r = model.predict(str(f), **kw)[0]
        cv2.imwrite(str(out_dir / f"{f.stem}_pred.jpg"), r.plot())
        dets = Counter(r.names[int(c)] for c in r.boxes.cls)
        confs = [round(float(c), 2) for c in r.boxes.conf]
        speed = {k: round(v, 1) for k, v in r.speed.items()}
        rows.append({"file": f.name, "shape": list(r.orig_shape), "detections": dict(dets), "conf": confs, "speed_ms": speed})
        print(f"[img] {f.name:30s} {dict(dets)} | speed ms {speed}")
    return rows


def run_video(model, src, name, kw, args, out_dir):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"LOI: khong mo duoc video/webcam: {src}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0
    fps_in = fps_in if 1 <= fps_in <= 240 else 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / f"{name}_pred.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    n, wall_ms, infer_ms = 0, [], []
    det_counts, track_ids = Counter(), defaultdict(set)
    frames_with_ids = 0
    t_start = time.perf_counter()
    while True:
        if args.max_frames and n >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        if args.track:
            r = model.track(frame, persist=True, tracker=args.tracker, **kw)[0]
        else:
            r = model.predict(frame, **kw)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        if n >= WARMUP_FRAMES:
            wall_ms.append(dt)
            infer_ms.append(r.speed["inference"])
        for c in r.boxes.cls:
            det_counts[r.names[int(c)]] += 1
        if args.track and r.boxes.id is not None:
            frames_with_ids += 1
            for c, i in zip(r.boxes.cls, r.boxes.id):
                track_ids[r.names[int(c)]].add(int(i))
        annotated = r.plot()
        writer.write(annotated)
        n += 1
        if args.show:
            cv2.imshow("stage3", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    total_s = time.perf_counter() - t_start
    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    mean_wall = sum(wall_ms) / len(wall_ms) if wall_ms else 0
    row = {
        "source": str(src),
        "frames": n,
        "resolution": [w, h],
        "fps_input": round(fps_in, 1),
        "model_ms_per_frame": stats(wall_ms),
        "inference_ms": stats(infer_ms),
        "model_fps": round(1000 / mean_wall, 1) if mean_wall else 0,
        "end_to_end_fps_incl_read_draw_write": round(n / total_s, 1) if total_s else 0,
        "detections_total": dict(det_counts),
        "output": str(out_path),
    }
    if args.track:
        row["unique_track_ids"] = {k: len(v) for k, v in track_ids.items()}
        row["frames_with_ids"] = frames_with_ids
    print(f"[vid] {name}: {json.dumps(row, ensure_ascii=False)}")
    return row


def main():
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        sys.exit("LOI: torch khong thay GPU.")
    if args.track:
        try:
            import lap  # noqa: F401
        except ImportError:
            sys.exit("LOI: tracking can package 'lap' (chua cai). Hoi truoc khi cai: pip install \"lap>=0.5.12\"")

    images, videos, used_assets = collect_sources(args.source)
    out_dir = ROOT / "runs" / "stage3" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    classes = None if args.classes == "all" else [int(c) for c in args.classes.split(",")]
    device = "cpu" if args.device == "cpu" else int(args.device)
    kw = dict(device=device, imgsz=args.imgsz, conf=args.conf, classes=classes, verbose=False)
    if args.half:
        kw["quantize"] = 16  # 'half' da deprecated o ultralytics 8.4
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    summary = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model": Path(args.model).name,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "half": args.half,
        "classes": [model.names[c] for c in classes] if classes else "all",
        "used_ultralytics_assets": used_assets,
        "images": [],
        "videos": [],
    }
    if images:
        summary["images"] = run_images(model, images, kw, out_dir)
    for v in videos:
        model.predictor = None  # reset tracker/predictor giua cac video
        if isinstance(v, int):  # webcam
            summary["videos"].append(run_video(model, v, f"webcam{v}", kw, args, out_dir))
        else:
            summary["videos"].append(run_video(model, str(v), v.stem, kw, args, out_dir))

    if summary["images"] or summary["videos"]:
        run_device = model.predictor.model.device
        summary["run_device"] = str(run_device)
    if torch.cuda.is_available():
        summary["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("-" * 60)
    print("run device :", summary.get("run_device"))
    print("VRAM peak  :", summary.get("vram_peak_mb"), "MB")
    print("ket qua    :", out_dir)


if __name__ == "__main__":
    main()
