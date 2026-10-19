import sys, time
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "| CUDA build:", torch.version.cuda)
if not torch.cuda.is_available():
    print("LOI: torch khong thay GPU (co the dang la ban CPU). Cai lai torch cu126.")
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
print("GPU    :", torch.cuda.get_device_name(0), "| VRAM %.1f GB" % (props.total_memory / 1024**3))

model = YOLO("models/yolo11n.pt")
src = str(ASSETS / "bus.jpg")

model.predict(src, device=0, imgsz=640, verbose=False)  # warm-up
torch.cuda.synchronize()
n = 20
t0 = time.perf_counter()
for _ in range(n):
    r = model.predict(src, device=0, imgsz=640, verbose=False)[0]
torch.cuda.synchronize()
ms = (time.perf_counter() - t0) * 1000 / n

dev = model.predictor.model.device  # predictor dung ban deepcopy cua model, khong phai model.model
print("model device:", dev, "| result tensor:", r.boxes.data.device)
print("detections  :", [(r.names[int(b.cls)], round(float(b.conf), 2)) for b in r.boxes])
print("speed ms (pre/infer/post):", {k: round(v, 1) for k, v in r.speed.items()})
print("total ms/anh (trung binh %d lan): %.1f" % (n, ms))
print("VRAM peak: %.0f MB" % (torch.cuda.max_memory_allocated() / 1024**2))
print("OK: YOLO chay tren GPU" if dev.type == "cuda" else "CANH BAO: model dang o CPU")
