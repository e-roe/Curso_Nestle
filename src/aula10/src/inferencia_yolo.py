import os
import cv2
from ultralytics import YOLO

bbox_colors = {'capacete': (255, 100, 100), 'colete': (255, 0, 255), 'abafador': (10, 250, 100)}

if __name__ == "__main__":
    images_dir = './data/test'

    # Load YOLO model
    model = YOLO('./runs/detect/train/weights/best.pt')

    # Supported image extensions
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp')

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(img_exts):
            continue
        img_path = os.path.join(images_dir, img_name) 
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Run inference
        results = model(img)
        for result in results:
            for box in result.boxes:
                boxc = box.xyxy.cpu().numpy()[0]
                color = bbox_colors[result.names[int(box.cls[0])]]
                cv2.rectangle(img, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), (0, 0, 0), 8)
                cv2.rectangle(img, (int(boxc[0]), int(boxc[1])), (int(boxc[2]), int(boxc[3])), color, 3)
                cv2.putText(img, result.names[int(box.cls[0])], (int(boxc[0]), int(boxc[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 0), 2)
                cv2.putText(img, result.names[int(box.cls[0])], (int(boxc[0]) + 3, int(boxc[1]) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 255), 2)
        cv2.imshow('', img)
        cv2.waitKey(0)
