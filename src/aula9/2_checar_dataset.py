import cv2
import sys
import os

bbox_colors = {'Ambulance': (255, 100, 100), 'Bus': (255, 0, 255), 'Car': (10, 250, 100),
               'Motorcycle': (100, 150, 255), 'Truck': (255, 0, 255)}


def check_annot_yolo7(root_path, classes, save_video=False):
    bbox_counter = 0
    frame_counter = 0
    files = os.listdir(os.path.join(root_path, 'labels'))
    for file in files:
        with open(os.path.join(root_path, 'labels', file)) as f:
            lines = f.readlines()
            if len(lines) == 0:
                print(f'Arquivo {file}, sem anotacoes')
                sys.exit()
        image = None

        for line in lines:
            ll = line[:].split(' ')
            if image is None:

                if image is None:
                    image = cv2.imread(os.path.join(root_path, 'images', file[:-3] + 'jpg'))
                frame_counter += 1

            width = image.shape[1]
            height = image.shape[0]
            cx = float(ll[1])
            cy = float(ll[2])
            w = float(ll[3])
            h = float(ll[4])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            color = bbox_colors[classes[int(ll[0])]]
            cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (255, 255, 255), 4)
            cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), color, 4)
            cv2.putText(image, classes[int(ll[0])],(int(x1 * width), int(y1 * height) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, color,
                        2, lineType=cv2.LINE_AA)
            cv2.putText(image, classes[int(ll[0])],(int(x1 * width) + 3, int(y1 * height) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255),
                        2, lineType=cv2.LINE_AA)
            bbox_counter += 1

        cv2.imshow('', image)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    CLASSES = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
    src_path = '../../datasets/Veiculos/test'
    check_annot_yolo7(src_path, CLASSES, save_video=True)

