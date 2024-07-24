import os
import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np


class IrisRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Recognition")

        self.file_path = tk.StringVar()

        self.setup_main_window()

        self.create_widgets()

        self.show_default_image()

    def setup_main_window(self):
        window_width = 1200
        window_height = 700

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        label_description = tk.Label(frame, text="Image File Path:", font=("Helvetica", 11))
        label_description.grid(row=0, column=0, pady=5, columnspan=2, padx=150, sticky="w")
        entry = tk.Entry(frame, width=50, textvariable=self.file_path, font=("Helvetica", 11), state="readonly")
        entry.grid(row=1, column=0, pady=10, columnspan=2)

        buttons_config = [
            {"text": "Insert Image", "command": self.browse_image, "width": 16, "height": 2, "row": 1, "column": 2},
            {"text": "Detect Iris", "command": self.detect_iris, "width": 20, "height": 2, "row": 2, "column": 0,
             "padx": 40},
            {"text": "Segment Iris", "command": self.segment_iris, "width": 20, "height": 2, "row": 2, "column": 2,
             "padx": 40},
            {"text": "Iris Recognition", "command": self.iris_recognition, "width": 20, "height": 2, "row": 3,
             "column": 1, "pady": 10},
        ]

        for button_config in buttons_config:
            button = self.create_button(frame, button_config)
            button.grid(row=button_config["row"], column=button_config["column"], padx=button_config.get("padx", 0),
                        pady=button_config.get("pady", 0))

        self.image_label = tk.Label(frame)
        self.image_label.grid(row=2, column=1, pady=20)

        self.result_text = tk.Text(frame, height=5, width=40, wrap="word", font=("Helvetica", 12))
        self.result_text.insert(1.0, "Iris Recognition Status...")
        self.result_text.grid(row=4, column=1, pady=20)

    def create_button(self, frame, config):
        return tk.Button(frame, text=config["text"], command=config["command"],
                         width=config["width"], height=config["height"],
                         padx=config.get("padx", 0), pady=config.get("pady", 0))

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            self.file_path.set(file_path)
            image = cv2.imread(file_path)
            self.show_image(image)

    def show_image(self, image):
        if image.shape[1] < 300:
            image = cv2.resize(image, (300, 300))
        else:
            image = cv2.resize(image, (image.shape[1], 300))

        photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def show_default_image(self):
        default_image_path = "resources/default_image.png"
        default_image = cv2.imread(default_image_path)
        default_image = cv2.resize(default_image, (400, 300))

        if default_image is not None:
            self.show_image(default_image)
        else:
            print("Eroare la încărcarea imaginii implicite.")

    def detect_outer_iris(self, image):
        outer_detection = cv2.GaussianBlur(image, (7, 7), 1)
        outer_detection = cv2.Canny(outer_detection, 20, 70, apertureSize=3)

        hough_circle = cv2.HoughCircles(outer_detection, cv2.HOUGH_GRADIENT, 1.3, 800)
        if hough_circle is not None:
            hough_circle = np.round(hough_circle[0, :]).astype("int")
            for (x, y, radius) in hough_circle:
                cv2.circle(image, (x, y), radius, (0, 128, 255), 4)

    def detect_inner_iris(self, image):
        inner_detection = cv2.GaussianBlur(image, (7, 7), 1)
        inner_detection = cv2.Canny(inner_detection, 100, 120, apertureSize=3)

        hough_circles = cv2.HoughCircles(inner_detection, cv2.HOUGH_GRADIENT, 1, 800,
                                         param1=50, param2=20, minRadius=0, maxRadius=60)
        circles = np.round(hough_circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 128, 255), 2)
            cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

    def detect_iris(self):
        file_path = self.file_path.get()

        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            output = image.copy()

            self.detect_outer_iris(output)
            self.detect_inner_iris(output)

            self.show_image(output)

    def segment_iris(self):
        file_path = self.file_path.get()

        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            output = image.copy()

            segmented_iris = self.segment_iris_algorithm(output)

            self.show_image(segmented_iris)

    def segment_iris_algorithm(self, image):
        image_blurred = cv2.GaussianBlur(image, (5, 5), 0)

        edges = cv2.Canny(image_blurred, 30, 60)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            max_radius_circle = max(circles, key=lambda x: x[2])

            reduced_radius = max_radius_circle[2] - 5 if max_radius_circle[2] > 5 else max_radius_circle[2]

            segmented_iris = np.zeros_like(image)

            cv2.circle(segmented_iris, (max_radius_circle[0], max_radius_circle[1]), reduced_radius, (255),
                       thickness=-1)
            segmented_iris = cv2.bitwise_and(image, segmented_iris)

            return segmented_iris

        return None

    def iris_recognition(self):
        file_path = self.file_path.get()

        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image2 = self.segment_iris_algorithm(image)
            inserted_image = cv2.GaussianBlur(image2, (7, 7), 1)
            inserted_image = cv2.Canny(inserted_image, 20, 70, apertureSize=3)

            threshold = 55
            access_granted = False

            folder_path = 'data_set'
            files = os.listdir(folder_path)
            image_files = [file for file in files if file.lower().endswith(('.jpg', '.jpeg'))]

            for image_file in image_files:
                image_to_compare_path = os.path.join(folder_path, image_file)
                image_to_compare_read = cv2.imread(image_to_compare_path, cv2.IMREAD_GRAYSCALE)
                image_to_compare_segmented = self.segment_iris_algorithm(image_to_compare_read)
                image_to_compare = cv2.GaussianBlur(image_to_compare_segmented, (7, 7), 1)
                image_to_compare = cv2.Canny(image_to_compare, 20, 70, apertureSize=3)

                sift = cv2.SIFT_create()
                kp_1, desc_1 = sift.detectAndCompute(inserted_image, None)
                kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

                index_params = dict(algorithm=0, trees=5)
                search_params = dict()
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                matches = flann.knnMatch(desc_1, desc_2, k=2)

                good_points = [m for m, n in matches if m.distance < 0.6 * n.distance]

                number_keypoints = min(len(kp_1), len(kp_2))

                if len(good_points) / number_keypoints * 100 >= threshold:
                    threshold = len(good_points) / number_keypoints * 100
                    best_good_points = good_points
                    best_kp_1 = kp_1
                    best_kp_2 = kp_2
                    best_image = image_to_compare
                    access_granted = True
                    image_class = image_file.split('.')[1]

            if access_granted:
                result_text = f"Keypoints 1ST Image: {len(best_kp_1)}\n" \
                              f"Keypoints 2ND Image: {len(best_kp_2)}\n" \
                              f"GOOD Matches: {len(best_good_points)}\n" \
                              f"How good it's the match: {threshold}%\n" \
                              f"Access Granted Class {image_class}"

                result = cv2.drawMatches(inserted_image, best_kp_1, best_image, best_kp_2, best_good_points, None)
                self.show_image(result)
            else:
                result_text = "Access Denied"

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = IrisRecognitionApp(root)
    root.mainloop()
