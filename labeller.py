import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import numpy as np
import os
class BoundingBoxApp:
    def __init__(self, master, image_path, initial_boxes=[], save_path=''):
        self.master = master
        self.image_path = image_path
        self.boxes = []  # Store bounding boxes as (rect_id, (x0, y0, x1, y1))
        self.selected_boxes = set()  # Track selected boxes for combining

        # Helper info
        # self.info_label = tk.Label(master, text="Right-click to select/deselect boxes, Left-click and drag to draw a new box.")
        # self.info_label.pack(side=tk.TOP)

        self.save_path = save_path
        # Load and display the image
        self.resize_w, self.resize_h = 960, 780
        self.image = Image.open(self.image_path).resize((self.resize_w, self.resize_h))
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(master, width=self.image.width, height=self.image.height)
        self.canvas.pack(side=tk.LEFT)

        self.canvas.image = self.photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Draw the initial bounding boxes
        for box_coords in initial_boxes:
            bounding_box = (
                int(box_coords[1] * self.image.width),
                int(box_coords[0] * self.image.height),
                int((box_coords[1] + box_coords[3]) * self.image.width),
                int((box_coords[0] + box_coords[2]) * self.image.height),
            )

            rect_id = self.canvas.create_rectangle(bounding_box, outline="green", width=2)

            self.boxes.append((rect_id, bounding_box))

        # Sidebar for buttons
        self.sidebar = tk.Frame(master, padx=5, pady=5)
        self.sidebar.pack(fill=tk.BOTH, side=tk.RIGHT)

        # Buttons
        self.delete_button = tk.Button(self.sidebar, text="Delete", command=self.delete_box)
        self.delete_button.pack(fill=tk.X)

        self.combine_button = tk.Button(self.sidebar, text="Combine", command=self.combine_boxes)
        self.combine_button.pack(fill=tk.X)

        self.confirm_button = tk.Button(self.sidebar, text="Confirm", command=self.confirm_and_exit)
        self.confirm_button.pack(fill=tk.X)

        # Mouse events for drawing
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.drawing)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        # Mouse event for selecting boxes
        self.canvas.bind("<ButtonPress-3>", self.select_box)

    def get_updated_boxes(self):
        updated_boxes = []

        for rect_id, _ in self.boxes:
            try:
                current_coords = self.canvas.coords(rect_id)
                # Ensure we have a valid list of coordinates
                if current_coords:
                    updated_coords = tuple(map(int, current_coords))
                    updated_boxes.append(updated_coords)
            except Exception as e:
                # Handle the case where the item doesn't exist or other errors
                print(f"Error retrieving coordinates for rect_id {rect_id}: {e}")
        return updated_boxes

    def start_draw(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_rectangle = None

    def drawing(self, event):
        if not self.start_x or not self.start_y:
            return

        if self.current_rectangle:
            self.canvas.delete(self.current_rectangle)

        self.current_rectangle = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y,
                                                              outline="green", width=2)

    def end_draw(self, event):
        self.boxes.append((self.current_rectangle, (self.start_x, self.start_y, event.x, event.y)))
        self.start_x = None

        self.start_y = None

        self.current_rectangle = None
    def add_box(self):
        # Simple dialog to enter box coordinates (for simplicity)
        coords = simpledialog.askstring("Box Coordinates", "Enter box coordinates (x0, y0, x1, y1):")
        if coords:
            x0, y0, x1, y1 = map(int, coords.split())
            rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="green", width=2)
            self.boxes.append((rect_id, (x0, y0, x1, y1)))

    def select_box(self, event):
        # Find all boxes that contain the click
        contained_boxes = []
        for rect_id, coords in self.boxes:
            x0, y0, x1, y1 = coords
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                box_area = (x1 - x0) * (y1 - y0)
                contained_boxes.append((rect_id, coords, box_area))

        # If there are no contained boxes, do nothing
        if not contained_boxes:
            return

        # Choose the box with the smallest area
        _, selected_coords, _ = min(contained_boxes, key=lambda x: x[2])

        # Find the rect_id for the selected_coords (there can be multiple, select the first one)
        selected_box = next((box for box in self.boxes if box[1] == selected_coords), None)

        if selected_box:
            rect_id, _ = selected_box
            # Toggle selection state
            if selected_box in self.selected_boxes:
                self.selected_boxes.remove(selected_box)
                self.canvas.itemconfig(rect_id, outline="green")  # Deselect visually
            else:
                self.selected_boxes.add(selected_box)
                self.canvas.itemconfig(rect_id, outline="blue")  # Select visually

    def confirm_and_exit(self):
        # Retrieve the updated bounding boxes
        updated_boxes = self.get_updated_boxes()

        normalized_bboxes = []
        boxes = []
        for each_original_bbox in updated_boxes:
            part_normalized_bbox = [
                                    each_original_bbox[1] / self.image.height,
                                    each_original_bbox[0] / self.image.width,

                                    (each_original_bbox[3] - each_original_bbox[1]) / self.image.height,
                                    (each_original_bbox[2] - each_original_bbox[0]) / self.image.width
                                    ]
            normalized_bboxes.append(part_normalized_bbox)

            boxes = [each_original_bbox[1],
                    each_original_bbox[0],
                    each_original_bbox[3],
                    each_original_bbox[2],
                    ]


        # save filtered bbox
        bboxes = {}
        bboxes['part_normalized_bbox'] = normalized_bboxes
        bboxes['bbox'] = boxes

        img_name = os.path.basename(self.image_path)[:-4]
        np.save(f"{self.save_path}/{img_name}.npy", bboxes)
        # Print to console (optional, for verification)
        print("Updated bounding boxes saved to 'grounding_dino/labels_manual/{img_name}.npy'")

        # Close the GUI window
        self.master.destroy()


    def delete_box(self):
        # Updated to delete all selected boxes
        for box in list(self.selected_boxes):
            self.canvas.delete(box[0])
            self.boxes.remove(box)
            self.selected_boxes.remove(box)

    def combine_boxes(self):
        # Combine all selected boxes
        if not self.selected_boxes:
            print("No boxes selected to combine.")
            return

        # Find min and max coordinates of all selected boxes
        min_x = min([coords[0] for _, coords in self.selected_boxes])
        min_y = min([coords[1] for _, coords in self.selected_boxes])
        max_x = max([coords[2] for _, coords in self.selected_boxes])
        max_y = max([coords[3] for _, coords in self.selected_boxes])

        # Create a new combined bounding box
        combined_rect_id = self.canvas.create_rectangle(min_x, min_y, max_x, max_y, outline="red", width=2)
        self.boxes.append((combined_rect_id, (min_x, min_y, max_x, max_y)))

        # Delete the individual selected boxes
        for box in list(self.selected_boxes):
            self.canvas.delete(box[0])
            self.boxes.remove(box)
        self.selected_boxes.clear()  # Clear the selection after combining

    def hover_box(self, event):
        # Change color of box on hover
        currently_hovered = None
        for rect_id, coords in self.boxes:
            x0, y0, x1, y1 = coords
            if x0 <= event.x <= x1 and y0 <= event.y <= y1:
                currently_hovered = rect_id
                if self.hovered_box != rect_id:
                    if self.hovered_box and self.hovered_box != self.selected_box:
                        self.canvas.itemconfig(self.hovered_box, outline="green")
                    self.hovered_box = rect_id
                    self.canvas.itemconfig(rect_id, outline="red")
                break
        if not currently_hovered and self.hovered_box and self.hovered_box != self.selected_box:
            self.canvas.itemconfig(self.hovered_box, outline="green")
            self.hovered_box = None


# # Example usage
# root = tk.Tk()
# app = BoundingBoxApp(root, 'images/IMG_0087.jpg', initial_boxes=[(10, 10, 100, 100), (150, 150, 250, 250)])
# root.mainloop()