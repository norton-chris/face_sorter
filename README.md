# Face Sorter
This repository contains Python code for a face sorter system, utilizing facial recognition technology along with image augmentation techniques to enhance accuracy.

## Getting Started
These instructions will provide a guide on how to replicate the setup of this project on your local machine for development and testing purposes.

## Prerequisites
Ensure that you have installed Python 3.x and conda on your machine. The Python libraries required for this project are listed in the requirements.txt file. To install these libraries, navigate to the project's root directory in your terminal and run:

```bash
pip install -r requirements.txt
```
To install the additional requirements with the conda environment, you can use the environment.yml file:

```bash
conda env create -f environment.yml
```
Then, activate the conda environment:

conda activate your_environment_name
Usage
The primary function of the system is process_file(file_path, output_dir). It takes a file path of an image and an output directory. The system then processes the image, detects faces, and if no faces are found, it applies augmentation transformations to improve the detection.

Through a GUI interface, the user can set the deep scan mode and initiate the deep scan process by pressing the "Deep Scan" button, which will trigger the transformations if no faces are initially detected.

## Features
Facial recognition to sort images with faces.
Image augmentation techniques applied when a face is not identified. Transformations include brightness changes, cropping, rotation, and upscaling.
User-friendly GUI for interaction.
How It Works
On loading an image, the system tries to detect faces. If unsuccessful, it enters a deep scan mode (if activated). In this mode, the system applies a set of transformations in a randomly selected order. Post each transformation, it attempts to detect faces again. This process continues until a face is identified or all transformations are exhausted. If a face is detected during this process, the system will continue with its normal face recognition process.

The image transformations leverage the imgaug library and consist of brightness alterations, cropping, rotation, and upscaling.

## Contributing
I welcome pull requests. For substantial changes, please open an issue first to discuss your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
