from setuptools import setup, find_packages

setup(
    name="ImageInsight",  # Name of your package
    version="0.1.0",  # Initial version
    packages=find_packages(),  # Automatically find packages in the current directory
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    install_requires=[
        "torch",              # PyTorch for neural networks
        "transformers",       # Hugging Face Transformers (for GPT-2 tokenizer)
        "Pillow",             # Pillow (for image manipulation)
        "numpy",              # NumPy (for numerical operations)
        "matplotlib",         # Matplotlib (for plotting)
        "torchvision",        # Torchvision (for image transformations and pre-trained models)
    ],

    entry_points={
        'console_scripts': [
            'my_package=my_package.main_script:main',  # Define your CLI entry point
        ],
    },
    author="Kinkini",
    author_email="kinkinimonaragala@gmail.com",
    description="A package to extract semantic activations from images using pre-trained models.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",  # Optional: link to the GitHub repository
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify supported Python versions
)