# Cat-Dog Image Classifier

A Python program that trains a Convolutional Neural Network (CNN) to classify images of cats and dogs.

## Overview

This program demonstrates how to build and train a deep learning model for image classification. It uses PyTorch, a popular deep learning framework, to create a neural network that can learn to distinguish between pictures of cats and dogs.

If you're new to machine learning, this readme will guide you through the concepts, requirements, and usage of this program.

## What is a CNN?

A Convolutional Neural Network (CNN) is a type of artificial neural network specifically designed for processing structured grid data like images. Here's how it works in simple terms:

1. **Convolutional Layers**: These scan the image with small filters to detect features like edges, textures, and patterns. Think of them as feature detectors that learn what parts of an image are important.

2. **Pooling Layers**: These reduce the image size while preserving important information. They help make the network more efficient and focus on what matters.

3. **Fully Connected Layers**: After extracting features with convolutional and pooling layers, these layers connect all the extracted features to make the final decision (cat or dog).

4. **Training Process**: The network initially makes random guesses, compares them to the correct answers, and gradually adjusts its internal parameters to make better predictions.

## Requirements

- Python 3.13 or higher
- PyTorch (1.7.0 or higher recommended)
- torchvision
- tqdm (for progress bars)
- Pillow (for image processing)
- GPU enabled machine (Apple or NVIDIA, optional but recommended for faster training)

You can install the required packages with:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The program expects your dataset to be organized in a specific way:

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

Each class (cat, dog) should have its own folder containing the relevant images.

## Basic Usage

Run the program with default parameters:

```bash
python main.py
```

This will:
1. Look for images in the `./data` directory
2. Resize all images to 256×256 pixels
3. Train for 10 epochs (complete passes through the dataset)
4. Save the trained model as `cat_dog_classifier.pth` and `cat_dog_classifier.onnx`

## Command Line Arguments

You can customize the training process with various arguments:

### Dataset Parameters:
- `--data_dir PATH`: Path to your dataset directory (default: './data')
- `--image_size SIZE`: Size to resize images to (default: 256)

### Training Parameters:
- `--batch_size SIZE`: Number of images to process at once (default: 32)
- `--learning_rate RATE`: Controls how quickly the model adapts (default: 0.001)
- `--num_epochs NUM`: Number of complete passes through the dataset (default: 10)
- `--momentum VAL`: Helps accelerate training in consistent directions (default: 0.9)
- `--weight_decay VAL`: Helps prevent overfitting (default: 1e-4)

### Output Parameters:
- `--model_path PATH`: Where to save the PyTorch model (default: 'cat_dog_classifier.pth')
- `--onnx_path PATH`: Where to save the ONNX model (default: 'cat_dog_classifier.onnx')

### Other Parameters:
- `--val_split RATIO`: Portion of data used for validation during training (default: 0.2)
- `--patience NUM`: Epochs to wait before reducing learning rate (default: 2)

## Example with Custom Parameters

```bash
python main.py --data_dir ./my_images --image_size 224 --batch_size 64 --num_epochs 20 --learning_rate 0.0005
```

## Understanding the Output

During training, you'll see progress bars and information about:

- **Loss**: How wrong the model's predictions are (lower is better)
- **Accuracy**: Percentage of correct predictions on the validation set
- **Learning Rate**: Current learning rate (may decrease during training)

At the end, the program saves:
- A PyTorch model (`.pth`) file: For use within other PyTorch applications
- An ONNX model (`.onnx`) file: For cross-platform deployment or use with other frameworks

## Machine Learning Concepts Explained

### Training vs. Validation
- **Training set**: The images the model learns from
- **Validation set**: Images kept separate to test how well the model generalizes

### Hyperparameters
- **Batch size**: Number of images processed at once (higher uses more memory but can be faster)
- **Learning rate**: Controls how quickly the model changes its parameters (too high can overshoot, too low can be slow)
- **Epochs**: Number of complete passes through the dataset (more epochs = more learning time)

### Overfitting
When a model performs well on training data but poorly on new data. Several techniques in this program help prevent overfitting:
- Dropout (randomly ignoring some neurons during training)
- Weight decay (penalizing large weights)
- Batch normalization (normalizing layer inputs)

## How to Use Your Trained Model

After training, you can use the model to classify new images. Here's a simple example:

```python
import torch
from torchvision import transforms
from PIL import Image
from cat_dog_classifier import CatDogCNN  # Import the model class

# Load the trained model
model = CatDogCNN(image_size=256)
model.load_state_dict(torch.load('cat_dog_classifier.pth'))
model.eval()  # Set to evaluation mode

# Prepare image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and transform an image
image = Image.open('new_image.jpg')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Make prediction
with torch.no_grad():
    output = model(input_tensor)
    
# Get class prediction
_, predicted = torch.max(output, 1)
class_names = ['cat', 'dog']
print(f'Prediction: {class_names[predicted.item()]}')
```

## Further Learning

If you're interested in learning more about machine learning and CNNs:

1. **PyTorch Tutorials**: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
2. **Convolutional Neural Networks**: [CS231n](http://cs231n.github.io/)
3. **Deep Learning Book**: Goodfellow, Bengio, and Courville, "Deep Learning" (2016)

## Troubleshooting

### Common Issues:

1. **Out of Memory Error**: Reduce batch size using `--batch_size`
2. **Slow Training**: Check if you're using GPU; if not, consider setting up CUDA
3. **Poor Accuracy**: Try training for more epochs, adjusting learning rate, or getting more training data
4. **Model Not Learning**: Ensure your dataset is correctly organized and contains sufficient examples

