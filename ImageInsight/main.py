# main.py
import torch
from transformers import GPT2Tokenizer
from .utils import process_semantic_activations
from .models import ActivationToDescriptionModel, extract_activations_from_images

class ImageInsight:
    def __init__(self, model_path, use_gpu=True):
        # Initialize tokenizer and model here
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = ActivationToDescriptionModel(activation_dim=4096, hidden_dim=256, vocab_size=self.tokenizer.vocab_size).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def run_pipeline(self, image_folder, model_name, layer_index, csv_output_path, csv_file_name):
        # Step 1: Extract activations from images
        X_test_tensor = extract_activations_from_images(
            model_name=model_name,
            layer_index=layer_index,
            image_folder_path=image_folder,
            use_gpu=(self.device.type == 'cuda'),
            csv_output_path=csv_output_path,
            csv_file_name=csv_file_name,
            image_extensions=['.jpg', '.png'],
        )

        # Step 2: Generate semantic activations and descriptions
        semantic_activations = process_semantic_activations(self.model, X_test_tensor, self.tokenizer, self.device)
        
        return semantic_activations
