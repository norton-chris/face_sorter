import os
import cv2
import face_recognition
import shutil
import pickle
from datetime import datetime
import csv

# The path of the CSV file to store the progress
progress_file = 'progress.csv'

# Create the progress file if it doesn't exist
if not os.path.exists(progress_file):
    with open(progress_file, 'w') as f:
        pass

# Load the set of directories that have already been scanned
try:
    with open(progress_file, 'r') as f:
        reader = csv.reader(f)
        scanned_directories = next(reader)
        # Prompt the user whether they want to continue from last time
        continue_scan = input("Do you want to continue from the last scan? (y/n): ")
        if continue_scan.lower() != 'y':
            scanned_directories = []
except StopIteration or FileNotFoundError:
    scanned_directories = []

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

# Process a directory of images
def process_directory(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # If this directory has already been scanned, skip it
        if root in scanned_directories:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                process_file(file_path, output_dir)

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

    # find faces in image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # draw a box around the face
        cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 0, 255), 2)

        # see if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # check if we have a match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            print("Face recognized:", name)
            cv2.putText(image_cv, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            cv2.imshow('Face', image_cv)
            while True:
                k = cv2.waitKey(0)
                if k == ord('c'):
                    break
                elif k == ord('x'):
                    cv2.destroyAllWindows()
                    new_name = input("Please enter the correct name of the person: ")
                    known_face_encodings.remove((name, face_encoding))
                    known_face_names.remove(name)
                    known_face_encodings.append((name, face_encoding))
                    known_face_names.append(new_name)
                    name = new_name
                    break
                elif k == ord('s'):
                    new_name = "Unknown"
                    if name == new_name:
                        break
                    known_face_encodings.remove(face_encoding)
                    known_face_names.remove(name)
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(new_name)
                    name = new_name
                    break
                elif k == ord('f'):
                    cv2.destroyAllWindows()
                    cv2.imshow('Face', cv2.resize(image_cv, (1920, 1080)))
                elif k == ord('r'):
                    cv2.destroyAllWindows()
                    cv2.imshow('Face', cv2.resize(image_cv, (640, 480)))

        else:
            cv2.imshow('Unrecognized Face', image_cv)
            while True:
                k = cv2.waitKey(0)
                if k == ord('s'):
                    name = "Unknown"
                    break
                elif k == ord('r'):
                    image_cv = cv2.resize(image_cv, (640, 480))
                    cv2.imshow('Unrecognized Face', image_cv)
                elif k == ord('f'):
                    image_cv = cv2.resize(image_cv, (1920, 1080))
                    cv2.imshow('Unrecognized Face', image_cv)
                else:
                    cv2.destroyAllWindows()
                    name = input("Please enter the name of the person: ")
                    cv2.destroyAllWindows()
                    break
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)

        # Get the directory for the name
        person_dir = os.path.join(output_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        # Get the directory for the month and year
        month_year_dir = os.path.join(person_dir, datetime.now().strftime('%B-%Y'))
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

    # Save face encodings and names
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f)

    with open('face_names.pkl', 'wb') as f:
        pickle.dump(known_face_names, f)


root_dir = "/Users/chris/Documents/Disney World 2011"#input("Enter the root directory: ")
output_dir = "/Users/chris/Documents/faces"
process_directory(root_dir, output_dir)
