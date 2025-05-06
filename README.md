# Colorizer

This project demonstrates how to colorize grayscale images using a deep learning model based on U-Net architecture. The model is trained to predict the `ab` channels of the LAB color space from grayscale input, using a U-Net neural network structure.

## Project Overview
The U-Net-based colorizer takes a grayscale image as input and generates a colorized version of the image. The model uses the LAB color space, where:
- The **L** channel represents lightness.
- The **a** and **b** channels represent the color components.

The network outputs the `ab` channels, which are combined with the original `L` channel to generate a fully colorized image.
## Requirements
The following Python libraries are required to run the project:

- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- tqdm

To set up the environment and install the required libraries, run the following command:
```bash
pip install -r requirements.txt
```

requirements.txt
```txt
torch>=1.7.1
opencv-python
numpy
matplotlib
tqdm
```

## Directory Structure
The following directory structure is used in this project:

```
Colorizer (mini project)
└── Colorizer
    ├── Dataset
    │   ├── data
    │   │   └── [Coloured Images].png
    │   ├── preprocessedimg
    │   │   ├── AB/
    │   │   │   └── [Images].npy
    │   │   └── L/
    │   │       └── [Images].png
    │   └── split/
    │       ├── train/
    │       │   ├── AB/
    │       │   └── L/
    │       ├── test/
    │       │   ├── AB/
    │       │   └── L/
    │       └── val/
    │           ├── AB/
    │           └── L/
    ├── models
    │   ├── checkpoints/
    │   │   ├── best_model.pth
    │   │   ├── checkpoint_epoch_5.pth
    │   │   └── training_history.png
    │   └── test results/
    │       └── [test].png
    └── source
        ├── preprocess.py
        ├── splitdata.py
        ├── train.py
        └── usemodel.py

```

## Model Description
The model architecture is based on the U-Net, a popular architecture for image-to-image tasks. It consists of:

- **Encoder:** A series of convolutional layers that downsample the input image.
- **Bottleneck:** A series of convolutional layers that perform the most abstract feature extraction.
- **Decoder:** A series of layers that upsample the image while combining features from the encoder via skip connections.

The output is the `ab` channels, predicted from the grayscale input (`L` channel).

## Usage
1. Load and Colorize a Single Image
To colorize a single image using the pretrained model, use the `colorize_and_display` function.

```python
colorize_and_display('path/to/grayscale_image.jpg')
```

This will load the grayscale image, run it through the model, and display the original and colorized images side by side.

2. Load and Colorize Multiple Images from a Directory
You can also colorize multiple images from a directory by modifying the `input_path` to point to a folder containing grayscale images.

```python
model_path = 'path/to/model/checkpoint.pth'
input_path = 'path/to/input/images'
output_path = 'path/to/output/directory'

# The colorized images will be saved to the output directory
```

This will process all images in the directory and save the colorized results.

3. Model Training
If you'd like to train your own colorizer model, you'll need to:

I. Prepare a dataset of grayscale and corresponding color images.

II. Modify and run train.py using your dataset (training script is not fully covered here but is adaptable).

Example
Below is an example usage for processing a single image:

```python
# Load model
model = UNetColorizer().to(device)
checkpoint = torch.load('path/to/model/checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint)

# Colorize image
image_path = 'path/to/grayscale_image.jpg'
colorized_image = colorize_and_display(image_path)

```

The original grayscale image and the colorized image will be displayed side by side.
### Model Checkpoint
The model checkpoint should be in the following format:

- A `.pth` file that contains the model weights, typically saved using `torch.save()`.

## Notes
- The model currently assumes the input image is in grayscale. Ensure that your images are in the correct format before using them for colorization.
- The U-Net architecture is designed to work on 256x256 input images. Input images will be resized to 256x256 before being processed by the model.

## Author
Parth Lathiya

~This is a part of one of mini projects.
