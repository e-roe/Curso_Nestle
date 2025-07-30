import os
import json
from PIL import Image

# Diretórios
images_dir = '../data/train'
labels_dir = '../data/train'
output_json = 'annotations.json'

images = []
annotations = []
ann_id = 1

# Defina suas categorias
categories = [
    {"id": 1, "name": "capacete"},
    {"id": 2, "name": "colete"},
    {"id": 3, "name": "abafador"}
]


def yolo_to_coco_bbox(bbox, img_w, img_h):
    x_center, y_center, w, h = bbox
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    return [x_min, y_min, w, h]


if __name__ == "__main__":
    print(f"Total de imagens: {len(images)}")
    print(f"Total de anotacoes: {len(annotations)}")
    print("Categorias:", [cat['name'] for cat in categories])

    # Verifica se os diretórios existem
    for idx, img_name in enumerate(os.listdir(images_dir)):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        if not os.path.exists(label_path):
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        images.append({
            "id": idx + 1,
            "file_name": img_name,
            "width": img_w,
            "height": img_h
        })

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0]) + 1  # YOLO classes geralmente começam em 0
                bbox = list(map(float, parts[1:]))
                coco_bbox = yolo_to_coco_bbox(bbox, img_w, img_h)
                annotations.append({
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                })
                ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(os.path.join(images_dir, output_json), 'w') as f:
        json.dump(coco_dict, f, indent=4)

    print(f"Anotacoes em COCO salvas em {output_json}")