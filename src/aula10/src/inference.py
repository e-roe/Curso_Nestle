import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import numpy as np

bbox_colors = {'Capacete': (255, 100, 100), 'Colete': (255, 0, 255), 'Abafador': (10, 250, 100)}

COCO_CLASSES = {0: "Background", 1: "Capacete", 2: "Colete", 3: "Abafador"}
# Initialize the model
num_classes = len(COCO_CLASSES)


# Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open image
    image_tensor = F.to_tensor(image).unsqueeze(0)  # Convert image to tensor and add batch dimension
    return image_tensor.to(device)


def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")


def draw_boxes_cv(image_path, prediction, window_name="Prediction", threshold=0.5):
    image = cv2.imread(image_path)
    image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = map(int, box)
            class_name = get_class_name(label)
            cor = bbox_colors[class_name]
            cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), cor, 2)
            cv2.putText(image_cv, f"{class_name} ({score:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    cv2.imshow(window_name, cv2.cvtColor(cv2.resize(image_cv, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Draw bounding boxes with the correct class names and increase image size
def draw_boxes(image, prediction, fig_size=(10, 10)):
    boxes = prediction[0]['boxes'].cpu().numpy()  # Get predicted bounding boxes
    labels = prediction[0]['labels'].cpu().numpy()  # Get predicted labels
    scores = prediction[0]['scores'].cpu().numpy()  # Get predicted scores

    # Set a threshold for showing boxes (e.g., score > 0.5)
    threshold = 0.5

    # Set up the figure size to control the image size
    plt.figure(figsize=fig_size)  # Adjust the figure size here

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)  # Get the class name
            plt.imshow(image)  # Display the image
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                              linewidth=2, edgecolor='r', facecolor='none'))
            plt.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r')

    plt.axis('off')  # Turn off axis
    plt.show()


if __name__ == "__main__":
    # Move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the trained model
    model = get_model(num_classes)
    model.load_state_dict(torch.load("../models/fasterrcnn_resnet50_epoch_20.pth"))
    model.to(device)
    model.eval()  # Set the model to evaluation mode


    # Load the unseen image
    path = "../data/test"
    images = os.listdir(path)
    for image in images:
        image_path = os.path.join(path, image)
        image_tensor = prepare_image(image_path)

        with torch.no_grad():  # Disable gradient computation for inference
            # `prediction` contains:
            # - boxes: predicted bounding boxes
            # - labels: predicted class labels
            # - scores: predicted scores for each box (confidence level)
            prediction = model(image_tensor)

            # Run the inference and display the results
            #draw_boxes(Image.open(image_path), prediction, fig_size=(12, 10))  # Example of increased size
            draw_boxes_cv(image_path, prediction)