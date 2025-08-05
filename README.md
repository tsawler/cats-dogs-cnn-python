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

Each class (cat, dog) should have its own folder containing the relevant images. You can use the Kaggle Cats and Dogs Dataset,
which can be [downloaded here](https://www.microsoft.com/en-us/download/details.aspx?id=54765).

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

### Early Stopping Parameters:
- `--early_stopping`: Enable early stopping to prevent overfitting
- `--early_stopping_patience NUM`: Number of epochs to wait before stopping if validation loss doesn't improve (default: 3)
- `--early_stopping_min_delta VAL`: Minimum change in validation loss to qualify as improvement (default: 0.001)

## Example with Custom Parameters

```bash
python main.py --data_dir ./my_images --image_size 224 --batch_size 64 --num_epochs 20 --learning_rate 0.0005
```

Example with early stopping enabled:
```bash
python main.py --early_stopping --early_stopping_patience 5 --num_epochs 50
```

This will stop training early if the validation loss doesn't improve for 5 consecutive epochs, even if it hasn't reached 50 epochs.

## Running Inference on a Single Image

After training a model, you can use it to classify individual images without retraining. The program supports inference mode for this purpose.

### Basic Inference Usage

To classify a single image using a trained model:

```bash
python main.py --inference --image_path path/to/your/image.jpg
```

This will:
1. Automatically look for a trained model (`cat_dog_classifier.pth` or `cat_dog_classifier.onnx`)
2. Load the image and preprocess it
3. Run the model to predict whether it's a cat or dog
4. Display the prediction and confidence score

### Inference Command Line Arguments

- `--inference`: Enable inference mode (required to run inference)
- `--image_path PATH`: Path to the image file you want to classify (required for inference)
- `--model_file PATH`: Specify a particular model file to use (optional, supports .pth or .onnx)
- `--image_size SIZE`: Image size the model was trained with (default: 256)

### Inference Examples

Using the default model:
```bash
python main.py --inference --image_path ./test_images/cat1.jpg
```

Using a specific PyTorch model:
```bash
python main.py --inference --image_path ./test_images/dog1.jpg --model_file ./models/best_model.pth
```

Using an ONNX model:
```bash
python main.py --inference --image_path ./test_images/cat2.jpg --model_file ./models/exported_model.onnx
```

Using a model trained with different image size:
```bash
python main.py --inference --image_path ./test_images/dog2.jpg --model_file custom_model.pth --image_size 224
```

### Inference Output

The inference mode will display:
- The path to the image being classified
- The predicted class (cat or dog)
- The confidence score as a percentage

Example output:
```
Using NVIDIA GPU (CUDA).
Using default PyTorch model: cat_dog_classifier.pth

Inference Results:
Image: ./test_images/fluffy_cat.jpg
Prediction: cat
Confidence: 96.52%
```

### Requirements for Inference

- A trained model file (.pth or .onnx format)
- For ONNX inference: `pip install onnxruntime`
- The same image size used during training (default: 256x256)

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
- Early stopping (stopping training when validation performance stops improving)

## How to Use Your Trained Model

After training, you have two options to classify new images:

### Option 1: Using the Built-in Inference Mode (Recommended)

The easiest way is to use the program's built-in inference mode:

```bash
python main.py --inference --image_path your_image.jpg
```

See the "Running Inference on a Single Image" section above for more details.

### Option 2: Using the Model in Your Own Python Code

You can also load the model in your own Python scripts:

```python
import torch
from torchvision import transforms
from PIL import Image
from main import CatDogCNN  # Import the model class

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
    
# Get class prediction with confidence
probabilities = torch.nn.functional.softmax(output, dim=1)
confidence, predicted = torch.max(probabilities, 1)
class_names = ['cat', 'dog']
print(f'Prediction: {class_names[predicted.item()]}')
print(f'Confidence: {confidence.item():.2%}')
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

## License
MIT license. See [LICENSE.md](LICENSE.md) for details.

## Acknowledgements

This program the uses the Kaggle Dogs vs. Cats imageset: Will Cukierski. Dogs vs. Cats. [https://kaggle.com/competitions/dogs-vs-cats](https://kaggle.com/competitions/dogs-vs-cats), 2013. Kaggle.