from ultralytics import YOLO
import os

# checkpointing
def save_checkpoint(trainer):
    epoch = trainer.epoch + 1
    save_epochs = {1, 25, 50, 75, 100}  # epoch
    if epoch in save_epochs:
        ckpt_dir = os.path.join(trainer.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        trainer.model[0].save(ckpt_path)
        print(f"\n✅ Checkpoint disimpan: {ckpt_path}")

if __name__ == "__main__":
    # model
    model = YOLO("ultralytics/cfg/models/11/yolo11mod.yaml")

    # callback
    model.add_callback("on_train_epoch_end", save_checkpoint)

    # training
    model.train(
        data="D:/TATATA/halo/dataset.yaml",
        epochs=100,
        imgsz=480,
        batch=2,
        workers=2,   # ubah ke 0 jika debugging di Windows
        device=0,
        project="D:/TATATA/halo/runs",
        name="yolo11mod_ripeness",
        save=True,   # tetap simpan best.pt & last.pt bawaan
    )
