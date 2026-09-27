# Fire Detection and Early Warning System in Complex Surveillance Scenes Based on Vision Large Models

A deep vision-based fire detection and early warning framework engineered for complex surveillance scenarios (e.g., industrial facilities, forest zones, underground utilities, and urban firefighting settings). Designed to address key real-world challenges including drastic lighting variations, dense background clutter, multi-scale targets, and alternating open flames or smoldering smoke.

This repository integrates vision large model backbones with high-efficiency detection architectures, offering a complete pipeline covering data preprocessing, model training, standard JSON prediction export, and an interactive Gradio Web UI.

---

## Key Features

* **Complex Environment Adaptability**: Specifically optimized with specialized architectures and robust data augmentation for weak-feature scenarios (e.g., intense lighting glare, small flame targets, smoldering smoke).
* **Cloud & Edge Deployment**: Supports PyTorch and TensorRT model export, enabling high-throughput and low-latency inference on cloud and edge computing nodes.
* **Standard Benchmark Integration**: Native compatibility with competition evaluation standards, automatically generating image-level `0/1` classification JSON files.
* **Interactive Visualization**: Built-in Gradio Web UI for real-time image and video stream detection, bounding box visualization, and confidence score outputs.

---

## Directory Structure

```text
.
├── configs/                # Model training and inference configurations
│   └── default_config.yaml
├── dataset/                # Dataset directory
│   ├── images/             # Training and evaluation images
│   ├── train_coco.json     # COCO-format object detection annotations
│   └── train_image.json    # Image-level classification annotations
├── models/                 # Model architectures and backbone extractions
│   ├── backbone.py
│   └── detector.py
├── utils/                  # Augmentation, metrics, and helper functions
│   ├── metrics.py          # Precision and Recall calculation logic
│   └── post_process.py
├── train.py                # Model training entry point
├── predict.py              # Batch inference & submission JSON generator
├── app.py                  # Gradio Web UI application
├── requirements.txt        # Project dependencies
└── README.md
```

---

## Environment Setup

Recommended environment: **Python 3.10+** and **CUDA 12.0+**.

```bash
# Clone the repository
git clone https://github.com/your-username/fire-detection-system.git
cd fire-detection-system

# Create and activate virtual environment
conda create -n fire_detect python=3.10 -y
conda activate fire_detect

# Install required packages
pip install -r requirements.txt
```

---

## Dataset Layout

Place training data inside the `dataset/` directory. Ensure your files follow this layout:

```text
dataset/
├── train_coco.json    # Object detection bounding boxes (COCO format)
├── train_image.json   # Image-level classification annotations (0/1)
└── images/            # Image directory
```

Sample structure for `train_image.json` (`0` indicates fire-free, `1` indicates fire present):

```json
{
  "frame_0001.jpg": 0,
  "frame_0002.jpg": 1
}
```

---

## Quick Start

### 1. Model Training

Launch model training using default or custom YAML configurations:

```bash
python train.py --config configs/default_config.yaml --batch-size 16 --epochs 50
```

### 2. Inference & Result Export

Run batch inference on the target test set to automatically produce the official prediction JSON format:

```bash
python predict.py --input-dir ./dataset/test_images --output-json submission.json --weights ./weights/best_model.pth
```

Sample output `submission.json`:

```json
{
  "test_001.jpg": 0,
  "test_002.jpg": 1
}
```

### 3. Interactive Web UI Deployment

Launch the Gradio Web interface to perform real-time detection and visualization:

```bash
python app.py --port 7860
```

Access `http://localhost:7860` in your browser to test images via drag-and-drop.

---

## Evaluation Metrics

Model performance is evaluated using image-level **Precision** and **Recall**. You can validate prediction outputs locally using:

```bash
python utils/metrics.py --gt dataset/train_image.json --pred submission.json
```

---

## Tech Stack & Ecosystem

This framework leverages modern vision architectures and open-source tooling:

* **Frameworks**: PyTorch / OpenMMLab / Ultralytics
* **Pre-trained Backbones**: DINOv3 / DEIM / YOLO series
* **Inference Engine**: TensorRT
* **Deployment & UI**: Gradio / ModelScope / Hugging Face