import warnings  # Import the warnings module

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # Import tqdm for progress bars

# --- Suppress specific Pillow warnings ---
# This will ignore UserWarnings related to "Truncated File Read"
# originating from the Pillow (PIL) library's TiffImagePlugin.
# While suppressing can make output cleaner, consider checking your image files
# if these warnings indicate actual data corruption.
warnings.filterwarnings("ignore", message="Truncated File Read", category=UserWarning, module="PIL.TiffImagePlugin")


# --- Configuration ---
# Set the path to your dataset. This assumes 'data' folder is in the same directory as the script.
DATA_DIR = './data'
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
IMAGE_SIZE = (128, 128) # Resize images to this resolution

# Check if CUDA (GPU) is available, then MPS (Apple GPU), otherwise use CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA GPU (CUDA).")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS).")
else:
    device = torch.device("cpu")
    print("Using CPU.")

# --- 1. Data Loading and Preprocessing ---
# Define transformations for the images
# - Resize: Resizes the image to a fixed size.
# - ToTensor: Converts a PIL Image or numpy.ndarray to a PyTorch tensor.
# - Normalize: Normalizes the tensor image with mean and standard deviation.
#              These values are common for ImageNet, good starting point.
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE), # Resize images to a consistent size
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize pixel values
])

# Load the dataset using ImageFolder.
# ImageFolder expects data organized in subdirectories where each subdirectory
# represents a class. E.g., data/cat/, data/dog/.
try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    print(f"Found {len(dataset)} images in total.")
    print(f"Classes: {dataset.classes}")
    print(f"Class to index mapping: {dataset.class_to_idx}")
except Exception as e:
    print(f"Error loading dataset from {DATA_DIR}. Please ensure the directory exists and contains 'cat' and 'dog' subdirectories with images.")
    print(f"Error details: {e}")
    # Exit or handle the error appropriately if the dataset isn't found
    exit()


# Split the dataset into training and validation sets
# We'll use 80% for training and 20% for validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoaders for batching and shuffling the data
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. Define the CNN Architecture ---
class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        # Convolutional layers
        # Input size: 3 channels (RGB) x 128 (height) x 128 (width)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Output: 32 x 128 x 128 (padding=1 keeps same size)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 32 x 64 x 64 (after pooling)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Output: 64 x 64 x 64
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 64 x 32 x 32

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # Output: 128 x 32 x 32
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 128 x 16 x 16

        # Fully connected layers
        # Calculate the input features for the first fully connected layer.
        # This depends on the output size of the last pooling layer: 128 channels * 16 * 16
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2) # 2 output classes: cat, dog

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        # The -1 means the dimension is inferred
        x = x.view(-1, 128 * 16 * 16)

        # Apply fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.fc2(x) # No activation here, as CrossEntropyLoss includes Softmax

        return x

# Instantiate the model and move it to the device (CPU/GPU)
model = CatDogCNN().to(device)
print("\nModel Architecture:")
print(model)

# --- 3. Loss Function and Optimizer ---
# CrossEntropyLoss is suitable for classification tasks. It combines LogSoftmax and NLLLoss.
criterion = nn.CrossEntropyLoss()
# Adam optimizer is a good general-purpose optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. Training Loop ---
print("\nStarting training...")
train_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train() # Set the model to training mode
    running_loss = 0.0
    # Wrap train_loader with tqdm for a progress bar
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Training)", unit="batch")
    for inputs, labels in train_loader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device) # Move data to device

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=running_loss/(train_loader_tqdm.n+1)) # Update loss in progress bar

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation Loop (after each epoch) ---
    model.eval() # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad(): # Disable gradient calculation for validation
        # Wrap val_loader with tqdm for a progress bar
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)", unit="batch")
        for inputs, labels in val_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # Get the class with the highest probability
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            val_loader_tqdm.set_postfix(accuracy=f"{100 * correct_predictions / total_samples:.2f}%")

    val_accuracy = 100 * correct_predictions / total_samples
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

print("\nTraining Finished!")

# --- 5. Save the trained model (optional) ---
MODEL_SAVE_PATH = 'cat_dog_classifier.pth'
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"PyTorch model saved to {MODEL_SAVE_PATH}")

# --- 6. ONNX Export ---
# Export the model to ONNX format for cross-platform deployment
ONNX_MODEL_PATH = 'cat_dog_classifier.onnx'
# Create a dummy input tensor with the same shape as your actual inputs
# This is necessary for ONNX export to trace the model's computation graph
dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device)

try:
    torch.onnx.export(model,                    # Trained model
                      dummy_input,              # A dummy input to trace the model
                      ONNX_MODEL_PATH,          # Where to save the ONNX model
                      export_params=True,       # Export all model parameters (weights and biases)
                      opset_version=11,         # ONNX opset version (common)
                      do_constant_folding=True, # Optimize model by folding constants
                      input_names = ['input'],  # Names for the input(s)
                      output_names = ['output'],# Names for the output(s)
                      dynamic_axes={'input' : {0 : 'batch_size'},    # Define dynamic batch size for input
                                    'output' : {0 : 'batch_size'}}) # Define dynamic batch size for output
    print(f"Model exported to ONNX format at {ONNX_MODEL_PATH}")
except Exception as e:
    print(f"Error during ONNX export: {e}")
    print("Please ensure you have the 'onnx' package installed (`pip install onnx`).")


# --- 7. Plotting Training Loss and Validation Accuracy ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, marker='o', color='green')
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Example of loading and using the model for inference ---
# You can uncomment and run this part after training to test the saved model.
"""
# Load the saved model (PyTorch .pth)
loaded_model = CatDogCNN().to(device)
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model.eval() # Set to evaluation mode

# Example inference (replace with a real image path)
# Make sure the image is in a path that exists
# For example, create a test image 'test_image.jpg'
# in the 'data/cat' or 'data/dog' folder or another test folder
test_image_path = os.path.join(DATA_DIR, 'cat', os.listdir(os.path.join(DATA_DIR, 'cat'))[0]) # Takes the first cat image for test
print(f"\nTesting with image: {test_image_path}")

try:
    from PIL import Image
    test_image = Image.open(test_image_path).convert('RGB') # Ensure RGB format
    test_tensor = transform(test_image).unsqueeze(0).to(device) # Add batch dimension and move to device

    with torch.no_grad():
        output = loaded_model(test_tensor)
        probabilities = torch.softmax(output, dim=1) # Get probabilities
        _, predicted_class_idx = torch.max(probabilities, 1)

    predicted_label = dataset.classes[predicted_class_idx.item()]
    confidence = probabilities[0, predicted_class_idx.item()].item() * 100

    print(f"Predicted: {predicted_label} with {confidence:.2f}% confidence.")
except FileNotFoundError:
    print(f"Error: Test image not found at {test_image_path}. Please provide a valid path.")
except Exception as e:
    print(f"Error during inference: {e}")
"""
