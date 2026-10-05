from ultralytics import YOLO

model =YOLO("C:/Users/ayush/OneDrive/Documents/prada_pro/coc/scripts/testing/best (1).pt")

print(model.names)
ckpt = model.ckpt
print("\nCheckpoint keys:", ckpt.keys())

# Sometimes version is stored here
if 'train_args' in ckpt:
    print("\nTrain args:", ckpt['train_args'])