from ultralytics import YOLO

# ---------------------------------------------------------------------------
# YOLO26 Instance Segmentation — House Exterior Elements
# (doors, windows, walls, roofs, etc.)
#
# Hardware : RTX 5060 Ti 16 GB VRAM · Ryzen 9 7900X (12C/24T)
# Dataset  : Roboflow export (YOLOv8/YOLO-seg format with data.yaml)
# ---------------------------------------------------------------------------

# ── Point this to your Roboflow dataset's data.yaml ──────────────────────────
DATASET_YAML = r"path/to/your/roboflow-dataset/data.yaml"

# ── Model selection ──────────────────────────────────────────────────────────
# yolo26s-seg  → fast + accurate enough for 423px architectural images (~22 MB)
# yolo26m-seg  → swap in if you want more accuracy and can tolerate ~2x slower
MODEL = "yolo26s-seg.pt"

# ── Hyper-parameters tuned for architectural segmentation ────────────────────
EPOCHS      = 100       # pre-augmented dataset converges faster; early stopping
                        # will halt sooner if the model plateaus
BATCH       = -1        # auto-batch: YOLO will find the largest batch that fits
                        # in 16 GB VRAM (expect ~64+ with yolo26s-seg at 448px)
IMGSZ       = 448       # nearest multiple of 32 ≥ 423; avoids wasting compute
                        # upscaling every image from 423 → 640
WORKERS     = 16        # 7900X has 24 threads; 16 workers keeps the GPU fed
                        # while leaving headroom for the OS and main process
CACHE       = "ram"     # cache all images in RAM — eliminates disk I/O bottleneck
                        # (use "disk" if you don't have enough system RAM)
PATIENCE    = 30        # stop early if val metric stalls for 30 epochs
OPTIMIZER   = "auto"    # YOLO26 defaults to MuSGD when "auto"
LR0         = 0.01
LRF         = 0.01      # cosine decay to lr0 * lrf
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3       # shorter warm-up since fewer total epochs
CLOSE_MOSAIC  = 10      # disable mosaic for the last 10 epochs

if __name__ == "__main__":
    model = YOLO(MODEL)

    results = model.train(
        data=DATASET_YAML,
        task="segment",
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        workers=WORKERS,
        cache=CACHE,
        patience=PATIENCE,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=CLOSE_MOSAIC,
        device=0,                   # GPU 0
        amp=True,                   # mixed precision — saves VRAM, speeds up training
        cos_lr=True,                # cosine LR schedule
        overlap_mask=True,          # allow overlapping masks (walls behind windows, etc.)
        mask_ratio=4,               # mask down-sample ratio (4 = default)
        plots=True,                 # save training curves & sample predictions
        save=True,                  # save best + last checkpoints
        save_period=25,             # also checkpoint every 25 epochs
        project="runs/segment",
        name="house-exterior",
    )

    # ── Validate on the best checkpoint ──────────────────────────────────────
    metrics = model.val()
    print(f"\nmAP50     : {metrics.seg.map50:.4f}")
    print(f"mAP50-95  : {metrics.seg.map:.4f}")

    # ── Quick inference sanity check ─────────────────────────────────────────
    # Uncomment to run prediction on a sample image after training:
    # preds = model.predict(source="path/to/test/image.jpg", save=True, conf=0.35)

    # ── Export to ONNX (optional, for deployment) ────────────────────────────
    # model.export(format="onnx", imgsz=448, simplify=True)
