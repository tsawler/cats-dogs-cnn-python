# Cat vs Dog Image Classifier

A PyTorch-based Convolutional Neural Network (CNN) for classifying images of cats and dogs.

## Overview

This project implements a CNN to distinguish between images of cats and dogs. The model is built with PyTorch, featuring:

- A 3-layer convolutional architecture
- Data preprocessing and augmentation
- Training with validation metrics
- Model export to both PyTorch (.pth) and ONNX formats
- Visualization of training performance

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- tqdm
- Pillow (PIL)
- onnx (for ONNX export)

Install dependencies:

```bash
pip install torch torchvision matplotlib tqdm Pillow onnx
```

## Dataset

### Getting the Dataset
You can download the cat and dog images dataset from Microsoft:
- Source: [Microsoft Cats and Dogs Dataset](https://www.microsoft.com/en-gb/download/details.aspx?id=54765)
- This dataset contains 25,000 images of dogs and cats for training machine learning algorithms

### Dataset Structure
After downloading, organize your dataset in the following structure:

```
data/
├── cat/
│   ├── cat_image1.jpg
│   ├── cat_image2.jpg
│   └── ...
└── dog/
    ├── dog_image1.jpg
    ├── dog_image2.jpg
    └── ...
```

The script expects a `data` directory in the same folder as the script with subdirectories for each class.

## Features

- **Device Flexibility**: Automatically uses NVIDIA GPU (CUDA), Apple GPU (MPS), or CPU based on availability
- **Data Preprocessing**: Resizes images to 128x128 and applies normalization
- **Training Progress**: Shows real-time progress bars with loss and accuracy metrics
- **Performance Visualization**: Plots training loss and validation accuracy over time
- **Model Export**: Saves the trained model in both PyTorch and ONNX formats for deployment

## Model Architecture

The CNN architecture consists of:

1. Three convolutional layers with ReLU activation and max pooling
2. Two fully connected layers
3. Output layer with 2 classes (cat and dog)

## Usage

### Training

Run the script to train the model:

```bash
python cat_dog_classifier.py
```

The script will:
1. Load and preprocess images from the data directory
2. Train the model for 10 epochs (configurable)
3. Display training progress with loss and validation accuracy
4. Save the trained model as `cat_dog_classifier.pth` and `cat_dog_classifier.onnx`
5. Generate performance plots

### Inference (Example)

The script includes commented code for inference that you can uncomment to test with your own images:

```python
# Example for using the trained model for inference
loaded_model = CatDogCNN().to(device)
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model.eval()

# Load and preprocess your test image
test_image = Image.open("path/to/your/test/image.jpg").convert('RGB')
test_tensor = transform(test_image).unsqueeze(0).to(device)

# Get prediction
with torch.no_grad():
    output = loaded_model(test_tensor)
    probabilities = torch.softmax(output, dim=1)
    _, predicted_class_idx = torch.max(probabilities, 1)

predicted_label = dataset.classes[predicted_class_idx.item()]
confidence = probabilities[0, predicted_class_idx.item()].item() * 100
print(f"Predicted: {predicted_label} with {confidence:.2f}% confidence.")
```

## Configuration

Modify these parameters at the top of the script to suit your needs:

```python
DATA_DIR = './data'       # Path to your dataset
BATCH_SIZE = 32           # Number of images per batch
LEARNING_RATE = 0.01      # Learning rate for optimizer
NUM_EPOCHS = 10           # Number of training epochs
IMAGE_SIZE = (128, 128)   # Image resolution
```

## Output Files

- `cat_dog_classifier.pth`: PyTorch model file
- `cat_dog_classifier.onnx`: ONNX model file for cross-platform deployment
- Training plots (displayed during execution)

## Notes

- The script includes warning suppression for common Pillow TiffImagePlugin warnings
- The dataset is split 80% for training and 20% for validation
- For best results, provide a diverse dataset with high-quality images