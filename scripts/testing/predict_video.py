"""
predict_video.py
----------------
Runs best.pt on a video and saves the annotated video.
"""

import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\models\yolo26n_craters_v1-5\weights\best.pt"
VIDEO_PATH = r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset\video.f399.mp4"
CONF_THRESH = 0.45

def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f"[INFO] Running inference on video: {VIDEO_PATH}")
    # Run prediction and save to output directory
    results = model.predict(source=VIDEO_PATH, conf=CONF_THRESH, save=True, project=r"C:\Users\ayush\OneDrive\Documents\prada_pro\coc\dataset", name="video_output", exist_ok=True)
    
    print(f"[INFO] Processing complete. Check the output directory for the saved video.")

if __name__ == "__main__":
    main()
