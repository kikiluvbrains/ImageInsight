import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import csv

## Visual
import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torchvision import transforms, models
import csv

def extract_activations_from_images(
    model_name='alexnet', 
    layer_index=4, 
    image_folder_path=None, 
    csv_output_path="/home/kinkini/Downloads/Graham_Movie",
    csv_file_name="visual_activations_output.csv",
    image_extensions=['.jpg', '.png'], 
    use_gpu=True
):
    """
    Extracts activations from a specified layer of a pre-trained model for images in a folder.

    Args:
        model_name (str): The name of the pre-trained model to use. Default is 'alexnet'.
        layer_index (int): The index of the layer to extract activations from. Default is 4.
        image_folder_path (str): The path to the folder containing the images.
        csv_output_path (str): The path where the CSV file will be saved.
        csv_file_name (str): The name of the CSV file to save. Default is 'visual_activations_output.csv'.
        image_extensions (list): List of file extensions for images. Default is ['.jpg', '.png'].
        use_gpu (bool): Whether to use GPU for computation. Default is True.

    Returns:
        np.array: Array containing the activations for each image.
    """

    # Load the pre-trained model
    model = getattr(models, model_name)(pretrained=True)
    model.eval()  # Set model to evaluation mode

    # Check if GPU is available and move the model to the correct device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Select the target layer
    if isinstance(model.classifier, nn.Sequential):
        target_layer = model.classifier[layer_index]
    else:
        raise ValueError("Layer index out of bounds for the selected model's classifier.")

    # Define preprocessing function
    def preprocess_image(image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        return input_tensor

    # Hook function to capture activations
    def get_activations(model, input_data, target_layer):
        activations = []

        def hook(module, input, output):
            activations.append(output)

        hook_handle = target_layer.register_forward_hook(hook)

        with torch.no_grad():
            model(input_data)

        hook_handle.remove()
        return activations[0]

    # Ensure the CSV output directory exists
    if not os.path.exists(csv_output_path):
        os.makedirs(csv_output_path)
    
    # Combine the output path and the file name to form the full CSV path
    csv_full_path = os.path.join(csv_output_path, csv_file_name)

    # Open CSV file to save activations
    with open(csv_full_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_name", "flattened_activation"])  # Write CSV header

        v_all_activations = []  # List to collect all activations

        # Collect activations from the images in the folder
        for root, dirs, files in sorted(os.walk(image_folder_path)):
            dirs.sort()
            # Filter and sort image files
            files = sorted([f for f in files if any(f.endswith(ext) for ext in image_extensions)])

            for filename in files:
                image_path = os.path.join(root, filename)
                print(f"Processing: {image_path}")
                input_data = preprocess_image(image_path)

                # Get activations for the chosen layer
                activations = get_activations(model, input_data, target_layer).cpu().numpy()
                print(f"Shape of activations for {filename}: {activations.shape}")

                # Flatten the activations
                flattened_activation = activations.flatten().tolist()

                # Append flattened activations to the list
                v_all_activations.append(flattened_activation)

                # Write the image name and flattened activations to the CSV file
                csv_writer.writerow([filename, flattened_activation])

        # Convert list of activations to a tensor (if needed)
        v_all_activations_tensor = torch.tensor(v_all_activations, dtype=torch.float32)

        print(f"Final shape of all activations tensor: {v_all_activations_tensor.shape}")

    return v_all_activations_tensor


## Semantic
# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token


class ActivationToDescriptionModel(nn.Module):
    def __init__(self, activation_dim, hidden_dim, vocab_size, dropout_prob=0.3):
        super(ActivationToDescriptionModel, self).__init__()
        self.fc1 = nn.Linear(activation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)  # Set bidirectional=True
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)  # Adjust the output layer

        # Adding dropout layers
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, target_sequence):
        # Map activation to hidden space
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first layer

        penultimate_activation = F.relu(self.fc2(x))
        penultimate_activation = self.dropout(penultimate_activation)  # Apply dropout after second layer

        # Repeat across the sequence length (which is fixed at 15)
        x = penultimate_activation.unsqueeze(1).repeat(1, target_sequence.size(1), 1)  # Shape: (batch_size, sequence_length, hidden_dim)

        # Decode the sequence
        decoder_output, _ = self.decoder(x)  # Shape: (batch_size, sequence_length, hidden_dim * 2)
        decoder_output = self.dropout(decoder_output)  # Apply dropout to decoder output

        output = self.fc_out(decoder_output)  # Shape: (batch_size, sequence_length, vocab_size)

        return output, penultimate_activation  # Return both output and penultimate layer activation
