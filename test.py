# Define paths and settings
image_folder = "/home/kinkini/Downloads/Graham_Movie/images_test"
model_name = 'alexnet'
layer_index = 4
use_gpu = True
csv_output_path="/home/kinkini/Downloads/Graham_Movie/visual_activations_output_try.csv",

# Initialize tokenizer (for description generation)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS

# Device setup
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

# Load your pre-trained activation to description model (ensure this model is defined)
model = ActivationToDescriptionModel(activation_dim=4096, hidden_dim=256, vocab_size=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load('/home/kinkini/Downloads/Graham_Movie/Scripts/best_model.pt', map_location=device))

# Step 1: Extract activations from images
X_test_tensor = extract_activations_from_images(
    model_name='alexnet',
    layer_index=4,
    image_folder_path=image_folder,
    use_gpu=False,  # Set to True if GPU is available
    csv_output_path="/home/kinkini/Downloads/Graham_Movie",
    csv_file_name="visual_activations_output.csv",
    image_extensions=['.jpg', '.png'], )

# Step 2: Generate semantic activations and descriptions
semantic_activations = process_semantic_activations(model, X_test_tensor, tokenizer, device)
