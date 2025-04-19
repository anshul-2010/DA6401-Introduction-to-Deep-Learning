# Video Object Detection using ImageAI

This sub-repository demonstrates how to perform object detection on video files using the [ImageAI](https://github.com/OlafenwaMoses/ImageAI) library. With minimal setup and code, you can run real-time or offline object detection using state-of-the-art models like YOLOv3.

## Features
- Detect objects in video files or webcam feeds.
- Built-in support for popular detection models like YOLOv3, RetinaNet, TinyYOLOv3.
- Easily customizable detection callbacks for frame-by-frame analysis.
- Save processed video with bounding boxes and object labels.

## Requirements

- Python 3.6+
- [ImageAI](https://github.com/OlafenwaMoses/ImageAI)
- OpenCV (`opencv-python`)
- TensorFlow or PyTorch (depending on the backend)
- Pre-trained YOLOv3 model (`yolov3.pt`)

---

### 1. Clone the Repository (Optional)
```bash
git clone https://github.com/your-username/video-object-detection.git
cd video-object-detection
```

### 2. Install Dependencies
```bash
pip install imageai opencv-python
```

You may also need:
```bash
pip install tensorflow  # or: pip install torch torchvision
```

### 3. Download YOLOv3 Model
Download the pre-trained YOLOv3 weights file from the official source:

- [YOLOv3 Model File (yolov3.pt)](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5)  
  *(Rename to `yolov3.pt` or update the filename in code accordingly)*

Place it in the root directory of your project.

## Usage

### Run Object Detection on Video
```python
from imageai.Detection import VideoObjectDetection
import os

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3.pt")
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path="traffic.mp4",
    output_file_path="traffic_detected",
    frames_per_second=20,
    log_progress=True
)
print(f"Processed video saved at: {video_path}")
```

### Custom Callback for Each Frame
```python
def forFrame(frame_number, output_array, output_count):
    print(f"Frame {frame_number}: Detected {output_count} objects")
    for obj in output_array:
        print(f"{obj['name']} - {obj['percentage_probability']:.2f}%")

detector.detectObjectsFromVideo(
    input_file_path="traffic.mp4",
    output_file_path="traffic_detected_callback",
    frames_per_second=20,
    per_frame_function=forFrame,
    log_progress=True
)
```

## Live Webcam Detection
```python
import cv2
camera = cv2.VideoCapture(0)

detector.detectObjectsFromVideo(
    camera_input=camera,
    output_file_path="live_detected",
    frames_per_second=20,
    log_progress=True,
    minimum_percentage_probability=50
)
```

## Output
- A new video file will be saved with detected objects overlaid.
- Each detected object will have a label and confidence score.
- Optionally, object counts per frame can be accessed using callbacks.

## Model Notes
This project uses the YOLOv3 model pre-trained on the COCO dataset, capable of detecting 80 classes including:
- Person
- Car
- Truck
- Dog
- Bicycle
- Traffic Light, etc.

## References
- [ImageAI Documentation](https://imageai.readthedocs.io/en/latest/)
- [COCO Dataset Classes](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/)
