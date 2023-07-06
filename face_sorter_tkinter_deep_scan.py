import os
import time
import random

import numpy as np

import cv2
import face_recognition
import shutil
import pickle
from datetime import datetime
import csv
import tkinter as tk
import threading

from FaceRecognizerGUI import FaceRecognizerGUI

# Create a class to represent your GUI
from PIL import ImageTk
from PIL import Image

# Some basic image transformations
def image_flip(image):
    return cv2.flip(image, 1)

def image_rotate(image, angle=90):
    height, width = image.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_img = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_img

def image_noise(image):
    noise = np.random.normal(0, 1, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    return image

def image_blur(image, kernel_size=(5,5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

transformations = [image_flip, image_rotate, image_noise, image_blur]

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
    gui.deep_scan.set(False)
    image = face_recognition.load_image_file(file_path)
    image_cv = cv2.imread(file_path)
    gui.show_image(image_cv)
    time.sleep(2)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image_cv, face_locations)


    if len(face_locations) == 0:
        print("No faces found, press deep scan if there are faces in the image")
        # Wait for 3 seconds for a deep_scan button press
        wait_for = 3
        for _ in range(3):
            if gui.deep_scan.get():
                break
            print("Press in", wait_for, flush=True)
            wait_for -= 1
            time.sleep(1)

        if gui.deep_scan.get():
            number_of_times_to_upsample = 0
            face_detected = False
            max_iterations = len(transformations)
            iteration = 0

            while not face_detected and iteration < max_iterations:
                # Apply random transformations
                random.shuffle(transformations)
                for transformation in transformations[:iteration + 1]:
                    print("Trying", transformation, "augmentation")
                    image_cv_transformed = transformation(image_cv)
                    gui.show_image(image_cv_transformed)

                    # Try face recognition on the transformed image
                    face_locations = face_recognition.face_locations(image_cv_transformed,
                                                                     number_of_times_to_upsample=number_of_times_to_upsample)
                    if face_locations:  # If a face is found, break the loop
                        face_detected = True
                        print("Face found with", transformation)
                        wait_for = 3
                        for _ in range(3):
                            if gui.deep_scan.get():
                                iteration = 0 # scan again
                                break
                            print("Press in", wait_for, flush=True)
                            wait_for -= 1
                            time.sleep(1)
                        break
                iteration += 1
                if number_of_times_to_upsample < 4:
                    number_of_times_to_upsample += 1

            # Proceed if a face has been detected
            if face_detected:
                face_encodings = face_recognition.face_encodings(image_cv, face_locations)
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Draw a box around the face
                    cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 0, 255), 2)
                    # Rest of your face recognition code...
            else:
                print("No faces detected even after transformations")

    face = False
    if face_locations is None:
        return face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face = True

        # draw a box around the face
        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 0, 255), 2)

        # see if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Calculate face distances
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) > 0:
            # Calculate confidence as 1 - normalized distance (this is just an example and is not standard)
            confidence_scores = [1 - (distance / max(face_distances)) for distance in face_distances]

            # Get the best match
            best_match_index = face_distances.argmin()
            best_match_score = confidence_scores[best_match_index]

            print(f"Best match score: {best_match_score * 100}%")
        else:
            print("No face distances found")

        #print(f"Best match score: {best_match_score * 100}%")

        name = "Unknown"

        # check if we have a match
        gui.show_image(image_cv)
        if True in matches and gui.run_autolabel.get():
            # If autolabeling is on, confirm the best match without waiting for manual confirmation
            name = known_face_names[best_match_index]
            print("Face automatically confirmed", name)
            cv2.putText(image_cv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            gui.show_image(image_cv)
            time.sleep(0.1)
        elif True in matches:
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
                    new_name = gui.name_entry.get()
                    gui.name_entry.delete(0, 'end')
                    try:
                        known_face_encodings.remove((name, face_encoding))
                        known_face_names.remove(name)
                        known_face_encodings.append((name, face_encoding))
                        known_face_names.append(new_name)
                    except ValueError:
                        pass
                    name = new_name
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
            time.sleep(0.5)
            while True and not gui.run_autolabel.get():
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
                    name = gui.name_entry.get()
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
    root_dir = "/Volumes/RAID_Drive/Family_Photos/"  # input("Enter the root directory: ")
    output_dir = "/Users/chris/Documents/faces"
    process_directory(root_dir, output_dir)
    root.quit()

# Use threading to run your existing script in a separate thread
script_thread = threading.Thread(target=run_script, args=())
script_thread.start()

# Start the Tkinter event loop in the main thread
root.mainloop()


