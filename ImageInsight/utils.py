import torch
import torch.nn.functional as F

def generate_description(model, activation, tokenizer, device, max_length=15):
    """
    Generate a description based on the given activation using the model's decoder.

    Args:
        model (nn.Module): The model used to generate the description.
        activation (torch.Tensor): The activation to generate a description from.
        tokenizer (GPT2Tokenizer): The tokenizer to convert tokens into words.
        device (torch.device): The device (CPU/GPU) to use for computation.
        max_length (int): Maximum length of the generated description.

    Returns:
        tuple: Generated description and the penultimate activations.
    """
    model.eval()  # Set model to evaluation mode
    activation = activation.to(device)  # Move activation to GPU/CPU

    # Start with a padding token as the initial input
    current_input = torch.tensor([[tokenizer.pad_token_id]], dtype=torch.long).to(device)

    generated_tokens = []
    
    # Generate the hidden state from the penultimate activation
    penultimate_activation = F.relu(model.fc2(F.relu(model.fc1(activation))))
    hidden_state = penultimate_activation.unsqueeze(0).repeat(2, 1, 1)  # For bidirectional GRU (2 directions)

    for _ in range(max_length):
        # Use the hidden state to predict the next token
        gru_input = penultimate_activation.unsqueeze(0).unsqueeze(1)  # Prepare input for GRU
        decoder_output, hidden_state = model.decoder(gru_input, hidden_state)
        logits = model.fc_out(decoder_output)[:, -1, :]  # Get logits for the last generated token
        
        next_token = torch.argmax(logits, dim=-1).item()  # Pick the token with the highest score
        generated_tokens.append(next_token)

        # Stop if the end of sequence (EOS) token is generated
        if next_token == tokenizer.eos_token_id:
            break

    # Decode the generated token sequence into a string
    generated_description = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_description, penultimate_activation


# Example usage with activations extracted
def process_semantic_activations(model, X_test_tensor, tokenizer, device):
    """
    Process the activations to generate semantic descriptions and return the penultimate activations.

    Args:
        model (nn.Module): The pre-trained model used for generating descriptions.
        X_test_tensor (torch.Tensor): Tensor of activations for the test images.
        tokenizer (GPT2Tokenizer): Tokenizer for decoding tokens into text.
        device (torch.device): The device to use for computation (CPU/GPU).

    Returns:
        list: A list of semantic activations and their descriptions.
    """
    semantic_activations = []

    for i, activation in enumerate(X_test_tensor):
        # Ensure the activation is moved to the correct device
        activation = activation.to(device)

        # Generate the description and extract penultimate activations
        generated_description, penultimate_activations = generate_description(model, activation, tokenizer, device)

        # Accumulate the penultimate activations for later analysis or usage
        semantic_activations.append({
            'description': generated_description,
            'penultimate_activations': penultimate_activations
        })

        # Print the generated description
        print(f"Generated Description for Activation {i + 1}: {generated_description}")

    return semantic_activations