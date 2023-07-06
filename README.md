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
```bash
conda activate your_environment_name
```

## Usage
### face_sorter.py
This script uses a command line interface. It is run from the terminal with the input directory and output directory as arguments. It reads images from the input directory, detects faces, and if no faces are found, it applies augmentation transformations to improve detection.

```bash
python face_sorter.py input_directory output_directory
```
### face_sorter_tkinter.py
This script provides a graphical user interface for easy use. When run, it opens a window that displays an image and a set of buttons for interacting with the image.

```bash
python face_sorter_tkinter.py
```
### face_sorter_tkinter_deep_scan.py
This script is similar to face_sorter_tkinter.py, but includes a deep scan feature. When activated, the deep scan feature applies a series of transformations to images that initially have no faces detected, to improve face detection.

```bash
python face_sorter_tkinter_deep_scan.py
```
### face_sorter_tkinter_tensorflow.py
This script uses TensorFlow for face recognition. The usage is similar to face_sorter_tkinter.py and face_sorter_tkinter_deep_scan.py.

```bash
python face_sorter_tkinter_tensorflow.py
```

The primary function of the system is process_file(file_path, output_dir). It takes a file path of an image and an output directory. The system then processes the image, detects faces, and if no faces are found, it applies augmentation transformations to improve the detection.

Through a GUI interface, the user can set the deep scan mode and initiate the deep scan process by pressing the "Deep Scan" button, which will trigger the transformations if no faces are initially detected.

## Features
Facial recognition to sort images with faces.  
User-friendly GUI for interaction.  
Image augmentation techniques are applied when a face is not identified. Transformations include brightness changes, cropping, rotation, and upscaling. (deep scan, in progress)  
Higher accuracy tensorflow detector (tensorflow program, in progress)  

## How It Works
All programs will ask if you'd like to resume at the unlabeled image. The progress is stored in ```progress.csv```. The face names and encodings are saved in ```face_names.pkl``` and ```face_encodings.pkl```

For Tkinter app, there are the following buttons:  
  &nbsp;**correct_name**: click this if the name recognized by the face recognition is correct.  
  &nbsp;**rename**: first, type the correct name in the text box and then click rename.  
  &nbsp;**Name as unknown**: names the person as Unknown.  
  &nbsp;**Deep Scan**: scans the image with varying augmentations (rotating, resizing, blurring, upscaling).  
  &nbsp;**Start Auto Label**: automatically label faces until Unknown face appears.  
  &nbsp;**Stop Auto Label**: stop automatically labeling faces.  

For Deep Scan, the system tries to detect faces. If unsuccessful, it enters a deep scan mode (if activated). In this mode, the system applies a set of transformations in a randomly selected order. Post each transformation, it attempts to detect faces again. This process continues until a face is identified or all transformations are exhausted. If a face is detected during this process, the system will continue with its normal face recognition process.

The image transformations leverage the imgaug library and consist of brightness alterations, cropping, rotation, and upscaling.

## Contributing
I welcome pull requests. For substantial changes, please open an issue first to discuss your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
