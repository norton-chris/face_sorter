import tkinter as tk
import cv2
from PIL import ImageTk
from PIL import Image


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
            'start': tk.Button(root, text="Start Auto Label", command=self.start_autolabel),
            'stop': tk.Button(root, text="Stop Auto Label", command=self.stop_autolabel),

        }

        # Place the buttons in the grid
        self.buttons['c'].grid(row=0, column=0)
        self.buttons['x'].grid(row=0, column=1)
        self.buttons['s'].grid(row=0, column=2)
        self.buttons['start'].grid(row=0, column=4)
        self.buttons['stop'].grid(row=0, column=5)
        self.run_autolabel = tk.BooleanVar(value=False)

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

    def start_autolabel(self):
        self.run_autolabel.set(True)

    def stop_autolabel(self):
        self.run_autolabel.set(False)