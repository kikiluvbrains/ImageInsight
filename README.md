# My Pipeline

A Python pipeline for extracting visual activations from images, processing them, and generating semantic descriptions using a neural network model. The pipeline utilizes PyTorch, a pre-trained AlexNet model, and a custom activation-to-description model.

## Features

- **Activation Extraction**: Extract visual activations from images using a pre-trained AlexNet model.
- **Semantic Descriptions**: Generate descriptions from the extracted activations using a custom neural network.
- **Device Support**: Optionally run on GPU or CPU.
- **Configurable Model**: Easily switch between different model layers for activation extraction.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Directory Structure](#directory-structure)
4. [Dependencies](#dependencies)
5. [License](#license)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/my_pipeline.git

2. Navigate into the project directory:

   ```bash
   cd my_pipeline

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt


## Usage

After installing the required dependencies, you can run the pipeline using your own set of images and a pre-trained model. Here's an example of how to use the pipeline:

```python
   from ImageInsight import ImageInsight  # Correct import of the ImageInsight class

   # Initialize the ImageInsight model with the path to the pre-trained model and GPU usage option
   insight = ImageInsight(model_path="path/to/your/model.pt", use_gpu=True)

   # Define paths and settings for running the pipeline
   image_folder = "path/to/your/images"  # Folder containing the input images
   model_name = "alexnet"  # Name of the pre-trained model to use
   layer_index = 4  # The index of the layer from which activations will be extracted
   use_gpu = False  # Set to True if a GPU is available for faster processing
   csv_output_path = "path/to/output/folder"  # Path to the folder where the CSV output will be saved
   csv_file_name = "visual_activations_output.csv"  # Name of the CSV file for the visual activations
   model_path = "path/to/your/model.pt"  # Path to the pre-trained model

   # Run the pipeline
   semantic_activations = insight.run_pipeline(
      image_folder=image_folder,
      model_name=model_name,
      layer_index=layer_index,
      csv_output_path=csv_output_path,
      csv_file_name=csv_file_name
   )

   # Print the generated semantic activations and descriptions
   print(semantic_activations)
```


## Directory Structure

   ```bash
   my_pipeline/
   │
   ├── my_pipeline/
   │   ├── __init__.py               # Package initialization
   │   ├── main.py                   # Main pipeline logic
   │   ├── models.py                 # Model definitions (e.g., ActivationToDescriptionModel)
   │   ├── utils.py                  # Utility functions (e.g., image activation extraction)
   │   └── tokenizer.py              # Tokenizer setup and handling
   │
   ├── README.md                     # Project documentation
   ├── requirements.txt              # Python dependencies
   ├── setup.py                      # Packaging information for pip
   ```

## Dependencies
   ```bash
   torch
   torchvision
   transformers
   Pillow
   numpy
   scikit-learn
   matplotlib
   ```


To install all dependencies, run:

```bash
pip install -r requirements.txt
```
## License

This project is licensed under the following terms:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, subject to the following conditions:

1. **Citation Requirement**: Any use of the Software in research or publications must cite the following GitHub repository:
   
   - [GitHub Repository Link](https://github.com/yourusername/ImageInsight)

2. **Attribution**: The above copyright notice, this permission notice, and the citation requirement must be included in all copies or substantial portions of the Software.

See the full [LICENSE](LICENSE) file for more details.


