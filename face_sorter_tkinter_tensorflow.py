import os
import time
import numpy as np

import cv2
import face_recognition
import shutil
import pickle
from datetime import datetime
import csv
import tkinter as tk
import threading

# Create a class to represent your GUI
from PIL import ImageTk
from PIL import Image

from tensorflow.keras.applications.resnet50 import preprocess_input

# Download the Caffe prototxt and weights files first
proto_path = 'deploy.prototxt.txt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

class FaceRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Face Recognizer')
        self.root.geometry('800x600')

        self.name_entry = tk.Entry(root)
        self.name_entry.grid(row=1, column=1)  # Place the entry in the grid

        # Use a Tkinter StringVar to store the button press
        self.key_pressed = tk.StringVar()
        self.buttons = {
            'c': tk.Button(root, text="Correct Name", command=lambda: self.key_pressed.set('c')),
            'x': tk.Button(root, text="Rename", command=lambda: self.key_pressed.set('x')),
            's': tk.Button(root, text="Name as Unknown", command=lambda: self.key_pressed.set('s')),
            'd': tk.Button(root, text="Deep Scan", command=lambda: self.deep_scan.set(True)),
        }

        # Place the buttons in the grid
        self.buttons['c'].grid(row=0, column=0)
        self.buttons['x'].grid(row=0, column=1)
        self.buttons['s'].grid(row=0, column=2)

        self.deep_scan = tk.BooleanVar(value=False)  # to check if deep scan is required

        # Place the buttons in the grid
        self.buttons['d'].grid(row=0, column=3)

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.grid(row=2, column=0, columnspan=3, sticky="nsew")  # This will allow the canvas to expand

        self.pil_image = None
        self.tk_image = None

        # Bind the Configure event (which occurs on window resize)
        self.canvas.bind("<Configure>", self.resize_image)

        # Configure the grid to resize properly
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

    def show_image(self, image):
        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a PIL image
        self.pil_image = Image.fromarray(image)

        self.resize_image()

    def resize_image(self, event=None):
        if self.pil_image:
            # Get the canvas size
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()

            # Keep the aspect ratio of the image
            image_aspect = self.pil_image.width / self.pil_image.height
            canvas_aspect = width / height

            if image_aspect < canvas_aspect:
                new_width = int(image_aspect * height)
                new_height = height
            else:
                new_width = width
                new_height = int(width / image_aspect)

            # Resize the image
            resized_image = self.pil_image.resize((new_width, new_height), Image.ANTIALIAS)

            # Convert the image to a Tkinter image
            self.tk_image = ImageTk.PhotoImage(resized_image)

            # Display the image on the canvas
            self.canvas.create_image(width / 2, height / 2, anchor=tk.CENTER, image=self.tk_image)

    def press_c(self):
        self.key_pressed = 'c'

    def press_x(self):
        self.key_pressed = 'x'
        self.new_name = self.name_entry.get()

    def press_s(self):
        self.key_pressed = 's'

    def press_f(self):
        self.key_pressed = 'f'

    def press_r(self):
        self.key_pressed = 'r'

# Create a new instance of the GUI
root = tk.Tk()
gui = FaceRecognizerGUI(root)

# The path of the CSV file to store the progress
progress_file = 'progress.csv'

if not os.path.exists(progress_file):
    with open(progress_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow([])

try:
    with open(progress_file, 'r') as f:
        reader = csv.reader(f)
        scanned_directories = next(reader)
        scanned_files = next(reader)
        # Prompt the user whether they want to continue from last time
        continue_scan = input("Do you want to continue from the last scan? (y/n): ")
        if continue_scan.lower() != 'y':
            scanned_directories = []
            scanned_files = []
except StopIteration or FileNotFoundError:
    scanned_directories = []
    scanned_files = []

# this list will hold the face encodings
known_face_encodings = []
# this list will hold the names corresponding to the encodings
known_face_names = []

# Load face encodings
if os.path.isfile('face_encodings.pkl'):
    with open('face_encodings.pkl', 'rb') as f:
        known_face_encodings = pickle.load(f)

# Load face names
if os.path.isfile('face_names.pkl'):
    with open('face_names.pkl', 'rb') as f:
        known_face_names = pickle.load(f)

# Write the set of directories that have been scanned to the CSV file
def save_progress():
    with open(progress_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(scanned_directories)
        writer.writerow(scanned_files)

# Process a directory of images
def process_directory(input_dir, output_dir):
    print("input dir:", input_dir, "len:", len(input_dir))
    for root, dirs, files in os.walk(input_dir):
        print("root:", root, "dir:", dirs, "files:", files)
        # If this directory has already been scanned, skip it
        image_count = 0
        if root in scanned_directories:
            print("Scanned photos:", image_count, end='\r')
            image_count += 1
            continue

        image_count = 0
        faceless_image_count = 0
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                if file_path in scanned_files:
                    continue  # Skip this file if it has already been processed
                print("Loading...", end="", flush=True)
                found_face = process_file(file_path, output_dir)
                image_count += 1
                if found_face:
                    print(f"\rFace image found at count: {image_count}, images since last face: {faceless_image_count}", end='', flush=True)
                    faceless_image_count = 0
                else:
                    print(f"\rFace image found at count: {image_count}, images since last face: {faceless_image_count}", end='', flush=True)
                    faceless_image_count += 1
                #print("\rCompleted!")
                # Mark this file as scanned
                scanned_files.append(file_path)
                save_progress()

        # Mark this directory as scanned
        scanned_directories.append(root)
        save_progress()

def is_image_file(file):
    img_extensions = ['.jpg', '.png', '.jpeg', '.bmp']
    ext = os.path.splitext(file)[-1]
    return ext.lower() in img_extensions

def process_file(file_path, output_dir):
    # load image
    image = face_recognition.load_image_file(file_path)
    image_cv = cv2.imread(file_path)
    (h, w) = image_cv.shape[:2]
    gui.show_image(image_cv)
    time.sleep(0.2)

    # find faces in image
    blob = cv2.dnn.blobFromImage(cv2.resize(image_cv, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        print("Face confidence:", confidence)
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (left, top, right, bottom) = box.astype("int")

            # Ensure the bounding boxes fall within the dimensions of the frame
            (left, top) = (max(0, left), max(0, top))
            (right, bottom) = (min(w - 1, right), min(h - 1, bottom))

            # extract face ROI and convert it from BGR to RGB
            face_image = image_cv[top:bottom, left:right]
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # find faces in image
            face_encodings = face_recognition.face_encodings(face_image_rgb)
    face = False
    for face_encoding in face_encodings:
        face = True
        # draw a box around the face
        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 0, 255), 2)

        # see if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Calculate face distances
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Calculate confidence as 1 - normalized distance (this is just an example and is not standard)
        confidence_scores = [1 - (distance / max(face_distances)) for distance in face_distances]

        # Get the best match
        best_match_index = face_distances.argmin()
        best_match_score = confidence_scores[best_match_index]

        print(f"Best match score: {best_match_score * 100}%")

        name = "Unknown"

        # check if we have a match
        gui.show_image(image_cv)
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            print("Face recognized:", name)
            cv2.putText(image_cv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            #cv2.imshow('Face', image_cv)
            # Display the image in the GUI
            gui.show_image(image_cv)
            while True:
                # Sleep for a small amount of time to reduce CPU usage
                time.sleep(0.1)
                #k = cv2.waitKey(0)
                if gui.key_pressed.get() == 'c':
                    print("Face confirmed")
                    break
                elif gui.key_pressed.get() == 'x':
                    print("Renaming face")
                    cv2.destroyAllWindows()
                    new_name = str(gui.name_entry)
                    gui.name_entry.delete(0, 'end')
                    try:
                        known_face_encodings.remove((name, face_encoding))
                        known_face_names.remove(name)
                        known_face_encodings.append((name, face_encoding))
                        known_face_names.append(new_name)
                    except ValueError:
                        pass
                    name = str(gui.name_entry)
                    break
                elif gui.key_pressed.get() == 's':
                    print('Naming Unknown')
                    new_name = "Unknown"
                    if name == new_name:
                        break
                    try:
                        known_face_encodings.remove((name, face_encoding))
                        known_face_names.remove(name)
                        known_face_encodings.append((name, face_encoding))
                        known_face_names.append(new_name)
                    except ValueError:
                        pass
                    name = new_name
                    break
                # elif gui.key_pressed == 'f':
                #     cv2.destroyAllWindows()
                #     cv2.imshow('Face', cv2.resize(image_cv, (1920, 1080)))
                # elif gui.key_pressed == 'r':
                #     cv2.destroyAllWindows()
                #     cv2.imshow('Face', cv2.resize(image_cv, (640, 480)))

        else:
            #cv2.imshow('Unrecognized Face', image_cv)
            # Display the image in the GUI
            print("Face unknown")
            cv2.putText(image_cv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            gui.show_image(image_cv)
            while True:
                if gui.key_pressed.get() == 's':
                    name = "Unknown"
                    print("Keeping as Unknown")
                    break
                # elif gui.key_pressed == 'r':
                #     image_cv = cv2.resize(image_cv, (640, 480))
                #     cv2.imshow('Unrecognized Face', image_cv)
                # elif gui.key_pressed == 'f':
                #     image_cv = cv2.resize(image_cv, (1920, 1080))
                #     cv2.imshow('Unrecognized Face', image_cv)
                elif gui.key_pressed.get() == 'x':
                    cv2.destroyAllWindows()
                    name = gui.name_entry
                    gui.name_entry.delete(0, 'end')
                    print("Naming face")
                    cv2.destroyAllWindows()
                    break
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

        # reset the variable for the next image
        gui.key_pressed.set('')
        gui.deep_scan.set(False)

        # Get the directory for the name
        person_dir = os.path.join(output_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        modification_time = os.path.getmtime(file_path)

        # Convert the modification time to a datetime object
        modification_datetime = datetime.fromtimestamp(modification_time)

        # Get the directory for the month and year
        month_year_dir = os.path.join(person_dir, modification_datetime.strftime('%b-%Y'))
        if not os.path.exists(month_year_dir):
            os.makedirs(month_year_dir)

        # Get the new filename
        base_filename = os.path.basename(file_path)
        filename, ext = os.path.splitext(base_filename)
        new_filename = f"{filename}_{'_'.join(known_face_names)}{ext}"
        new_file_path = os.path.join(month_year_dir, new_filename)

        # Copy the file
        shutil.copy2(file_path, new_file_path)
        print("file copied from:", file_path, "to", new_file_path)
        return face

    # Save face encodings and names
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f)

    with open('face_names.pkl', 'wb') as f:
        pickle.dump(known_face_names, f)

# Use threading to run the GUI and your existing script in parallel
# def run_gui():
#     root.mainloop()

def run_script():
    root_dir = "/Users/chris/Documents/Disney World 2011"  # input("Enter the root directory: ")
    output_dir = "/Users/chris/Documents/faces"
    process_directory(root_dir, output_dir)

# Use threading to run your existing script in a separate thread
script_thread = threading.Thread(target=run_script, args=())
script_thread.start()

# Start the Tkinter event loop in the main thread
root.mainloop()


