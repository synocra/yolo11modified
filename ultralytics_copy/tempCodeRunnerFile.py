from ultralytics import YOLO

# pastikan file YAML ada di path ini:
model = YOLO("ultralytics/cfg/models/11/yolo11mod.yaml")

# training
model.train(
    data="fruit.yaml",   # path dataset
    epochs=100,
    imgsz=640,
)
