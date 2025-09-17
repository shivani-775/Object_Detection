🖼️ Object Detection using YOLOv5

This project implements Object Detection using the YOLOv5 model.
It can detect and classify objects in images or videos with high accuracy.

📌 Features

Detects multiple objects in images and videos

Uses YOLOv5s pretrained weights (COCO dataset)

Runs on CPU or GPU (CUDA)

Easy to run with a single command

Custom dataset training support

🛠️ Tech Stack

Python 3.9+

PyTorch

YOLOv5

OpenCV

Numpy

📂 Project Structure
object-detection/
│── data/              # Images and datasets
│── runs/              # Detection results
│── models/            # YOLOv5 model architecture
│── detect.py          # Main detection script
│── requirements.txt   # Python dependencies
│── README.md          # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2️⃣ Create and activate virtual environment
# Create venv
python -m venv venv

# Activate venv
# Windows (PowerShell)
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

🚀 How to Run
Run object detection on sample images
python detect.py --source data/images --weights yolov5s.pt

Run detection on a video
python detect.py --source data/video.mp4 --weights yolov5s.pt

Run detection using webcam
python detect.py --source 0 --weights yolov5s.pt


Results will be saved in the runs/detect/ folder.
