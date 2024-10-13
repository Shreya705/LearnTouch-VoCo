import cv2
from ultralytics import YOLO

# create a list of all the items in the coco dataset

coco_dict = {
 0: u'person',
 1: u'bicycle',
 2: u'car',
 3: u'motorcycle',
 4: u'airplane',
 5: u'bus',
 6: u'train',
 7: u'truck',
 8: u'boat',
 9: u'traffic light',
 10: u'fire hydrant',
 11: u'stop sign',
 12: u'parking meter',
 13: u'bench',
 14: u'bird',
 15: u'cat',
 16: u'dog',
 17: u'horse',
 18: u'sheep',
 19: u'cow',
 20: u'elephant',
 21: u'bear',
 22: u'zebra',
 23: u'giraffe',
 24: u'backpack',
 25: u'umbrella',
 26: u'handbag',
 27: u'tie',
 28: u'suitcase',
 29: u'frisbee',
 30: u'skis',
 31: u'snowboard',
 32: u'sports ball',
 33: u'kite',
 34: u'baseball bat',
 35: u'baseball glove',
 36: u'skateboard',
 37: u'surfboard',
 38: u'tennis racket',
 39: u'bottle',
 40: u'wine glass',
 41: u'cup',
 42: u'fork',
 43: u'knife',
 44: u'spoon',
 45: u'bowl',
 46: u'banana',
 47: u'apple',
 48: u'sandwich',
 49: u'orange',
 50: u'broccoli',
 51: u'carrot',
 52: u'hot dog',
 53: u'pizza',
 54: u'donut',
 55: u'cake',
 56: u'chair',
 57: u'couch',
 58: u'potted plant',
 59: u'bed',
 60: u'dining table',
 61: u'toilet',
 62: u'tv',
 63: u'laptop',
 64: u'mouse',
 65: u'remote',
 66: u'keyboard',
 67: u'cell phone',
 68: u'microwave',
 69: u'oven',
 70: u'toaster',
 71: u'sink',
 72: u'refrigerator',
 73: u'book',
 74: u'clock',
 75: u'vase',
 76: u'scissors',
 77: u'teddy bear',
 78: u'hair drier',
 79: u'toothbrush'
}

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Open video capture
cap = cv2.VideoCapture(0)  # 0 represents the default camera

while True:
    # Read frame from video capture
    ret, frame = cap.read()

    # Run inference on the frame
    results = model(frame)

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs

        # Print the predictions
        print("Predicted Classes:", [coco_dict[i] for i in boxes.cls.tolist()])
        print("Masks:", masks)
        print("Keypoints:", keypoints)
        print("Probs:", probs)

        # Draw bounding boxes on the frame
        for box, class_name in zip(boxes.xyxy, [coco_dict[i] for i in boxes.cls.tolist()]):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()