# remove replicated ones
import numpy as np
import cv2
import os
from PIL import Image
import PIL
import glob
def calculate_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area.
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


# def nms(boxes, threshold=0.9):
#     filtered_boxes = []
#
#     for i in range(len(boxes)):
#         keep = True
#         for j in range(len(boxes)):
#             if i != j:
#                 iou = calculate_iou(boxes[i], boxes[j])
#                 if iou > threshold:
#                     keep = False
#                     break
#         if keep:
#             filtered_boxes.append(boxes[i])
#
#     return filtered_boxes
def calculate_area(box):
    # Calculate the area of a bounding box
    return (box[2] - box[0]) * (box[3] - box[1])

def is_not_too_small(box, min_dim=150):
    # Check if the box is not too small
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width >= min_dim or height >= min_dim
def is_small(box, min_dim=10):
    # Check if the box is not too small
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width <= min_dim or height <= min_dim

def is_too_small(box, min_dim=50):
    # Check if the box is not too small
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width <= min_dim or height <= min_dim

def contains_or_nearly_contains(boxA, boxB):
    # Check if boxA contains or nearly contains boxB
    return (boxA[0] <= boxB[0] and boxA[1] <= boxB[1] and boxA[2] >= boxB[2] and boxA[3] >= boxB[3])


# def filter_containing_boxes(boxes):
#     # Sort boxes by area
#     boxes.sort(key=calculate_area)
#     keep_indices = set(range(len(boxes)))  # Start with all indices marked to keep
#
#     # Iterate over the boxes
#     for i, boxA in enumerate(boxes):
#         for j in range(i + 1, len(boxes)):
#             if j not in keep_indices:
#                 continue
#             if (calculate_iou(boxA, boxes[j])>0.3 and is_not_too_small(boxes[j])):
#                 keep_indices.remove(i)  # Remove the index of the larger box
#
#
#     # Reconstruct the list of boxes to keep
#     filtered_boxes = [boxes[i] for i in keep_indices]
#     return filtered_boxes
def remove_biggest(boxes):
    def is_inside(inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    keep_indices = set(range(len(boxes)))
    for i, boxA in enumerate(boxes):
        num_inside = 0
        if (boxA[2] - boxA[0])>480 or (boxA[3] - boxA[1])>480:

            for j, boxB in enumerate(boxes):
                if i!=j:
                    if is_inside(boxB, boxA) and is_not_too_small(boxB):
                        num_inside+=1
            if num_inside>1:
                keep_indices.discard(i)
    filtered_boxes = [boxes[i] for i in keep_indices]
    return filtered_boxes


def remove_base(bboxes):
    keep_indices = set(range(len(bboxes)))  # Start with all indices marked to keep
    # Check every box against all others
    # remove the box that contain a big bbox (non-handle)
    for i, boxA in enumerate(bboxes):
        for j, boxB in enumerate(bboxes):
            if i != j and j in keep_indices:
                if is_inside(boxB, boxA):
                    if calculate_area(boxB)>200*200:
                        keep_indices.discard(i)
                        break  # Since boxA is removed, no need to compare it further

    # Reconstruct the list of boxes to keep
    filtered_boxes = [bboxes[i] for i in keep_indices]
    return filtered_boxes
def filter_containing_boxes(boxes):
    # No need to sort by area if we're not using area as a criterion for removal
    keep_indices = set(range(len(boxes)))  # Start with all indices marked to keep

    # Check every box against all others
    for i, boxA in enumerate(boxes):
        if i not in keep_indices:
            # Skip boxes that have already been removed
            continue

        for j, boxB in enumerate(boxes):
            if i != j and j in keep_indices:
                # Check if one box is larger, the other is not too small, and they have significant overlap
                overlap = calculate_iou(boxA, boxB)
                areaA = calculate_area(boxA)
                areaB = calculate_area(boxB)

                if overlap > 0.2:
                    # # If boxB is larger but not too small, and there is significant overlap, remove it
                    # if areaA < areaB and is_not_too_small(boxA):
                    #     keep_indices.discard(j)
                    # If boxA is larger but not too small, and there is significant overlap, remove it
                    if areaB < areaA and is_not_too_small(boxB):
                        keep_indices.discard(i)
                        break  # Since boxA is removed, no need to compare it further

    # Reconstruct the list of boxes to keep
    filtered_boxes = [boxes[i] for i in keep_indices]
    return filtered_boxes

def remove_handle_overlap(boxes, min_handle_size=100, overlap_threshold=0.1):
    # First, we sort the boxes based on area, so smaller boxes are at the beginning
    boxes.sort(key=calculate_area)

    filtered_boxes = []
    removed_indices = set()

    # Iterate over the boxes
    for i, boxA in enumerate(boxes):
        # If the box has already been removed due to overlap, skip it
        if i in removed_indices:
            continue

        # Assume the box is to be kept until we find an overlap
        keep = True

        # If the box is small enough to be a handle, check for overlaps with other handles
        if calculate_area(boxA) < (min_handle_size * min_handle_size):
            for j, boxB in enumerate(boxes[i + 1:], start=i + 1):
                # If Box B is a handle and overlaps with Box A significantly
                if calculate_area(boxB) < (min_handle_size * min_handle_size) and calculate_iou(boxA, boxB) > overlap_threshold:
                    # Mark Box B for removal
                    removed_indices.add(j)
                    # Box A is kept, so we can break the inner loop
                    break
        # If there were no disqualifying overlaps, add the box to the filtered list
        if keep:
            filtered_boxes.append(boxA)

    return filtered_boxes

def remove_small(bboxes):
    # First, we sort the boxes based on area, so smaller boxes are at the beginning
    bboxes.sort(key=calculate_area)

    filtered_boxes = []
    removed_indices = set()

    # Iterate over the boxes
    for i, boxA in enumerate(bboxes):
        # If the box has already been removed due to overlap, skip it
        if i in removed_indices:
            continue

        # Assume the box is to be kept until we find an overlap
        keep = True
        # If the box is small enough to be a handle, check for overlaps with other handles
        if calculate_area(boxA) < (10 * 10):
            keep=False
            # for j, boxB in enumerate(bboxes[i + 1:], start=i + 1):
            #     # If Box B is a handle and overlaps with Box A significantly
            #     if calculate_area(boxB) < (150 * 150) and calculate_iou(boxA, boxB) > 0.2 and is_too_small(bbox):
            #         # Mark Box B for removal
            #         removed_indices.add(j)
            #         # Box A is kept, so we can break the inner loop
            #         break
        # If there were no disqualifying overlaps, add the box to the filtered list
        if keep:
            filtered_boxes.append(boxA)

    return filtered_boxes

def nms(boxes, iou_threshold=0.75):
    filtered_boxes = filter_containing_boxes(boxes)
    filtered_boxes = remove_handle_overlap(filtered_boxes)
    # filtered_boxes = remove_biggest(filtered_boxes)
    filtered_boxes = remove_base(filtered_boxes)
    # filtered_boxes = remove_small(filtered_boxes)
    kept_boxes = []

    for box in filtered_boxes:
        # Assume this box is not overlapping significantly with any kept box
        keep = True
        for kept_box in kept_boxes:
            if calculate_iou(box, kept_box) >= iou_threshold:
                # If it overlaps significantly, we don't keep this box
                keep = False
                break
        if keep:
            kept_boxes.append(box)

    return kept_boxes


def is_inside(inner, outer):
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]
def add_handle_if_needed(boxes, min_dim=50):
    # Function to determine if one box is inside another


    # Function to add a handle box in the center
    def add_center_handle(box, handle_size=20):
        center_x = box[0] + (box[2] - box[0]) // 2
        center_y = box[1] + (box[3] - box[1]) // 2
        half_size = handle_size // 2
        return [center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size]

    new_boxes = boxes.copy()  # Copy the list to avoid modifying the original list

    for box in boxes:
        # Check if the box is not small
        if is_small(box, min_dim):
            continue

        # Check if no other boxes are inside this box
        if not any(is_inside(other_box, box) for other_box in boxes if other_box != box):
            # Check if the width is greater than the height
            if (box[2] - box[0]) < (box[3] - box[1]):
                # Add a handle box in the center
                handle_box = add_center_handle(box)
                new_boxes.append(handle_box)

    return new_boxes


def visualize_bbox(image, save_path, bboxes):
    for bounding_box in bboxes:
        start_point = (bounding_box[1], bounding_box[0])  # Top left corner
        end_point = (bounding_box[3], bounding_box[2])  # Bottom right corner
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 3  # Line thickness
        # Draw the rectangle
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imwrite(f"{save_path}.png", image)

def summary_kitchen(label_global_dir, label_part_dir, img_dir, save_dir):
    for img_path in glob.glob(img_dir+"/*"):
        img_name = os.path.basename(img_path)[:-4]
        object_info = np.load(f"{label_global_dir}/{img_name}.npy", allow_pickle=True).item()
        global_bbox_pred = object_info['part_normalized_bbox']  # assuming all images are normalize to 512x512
        image = cv2.imread(img_path)
        img_w, img_h = image.shape[0], image.shape[1]
        # PIL.Image.fromarray(image).show()
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        new_bboxes = []
        for each_bbox in global_bbox_pred:
            bounding_box = [int(each_bbox[0] * img_w),
                            int(each_bbox[1] * img_h),
                            int((each_bbox[0] + each_bbox[2]) * img_w),
                            int((each_bbox[1] + each_bbox[3]) * img_h),
                            ]
            new_bboxes.append(bounding_box)
        # global_bbox = nms(new_bboxes, iou_threshold=0.9)
        normalized_global_bboxes = global_bbox_pred
        normalized_part_bboxes = []
        part_bboxes = []
        for bbox_id, each_original_bbox in enumerate(global_bbox_pred):
            # process part boxes
            part_info = np.load(f"{label_part_dir}/{img_name}_{bbox_id}.npy", allow_pickle=True).item()
            part_bbox_pred = part_info['part_normalized_bbox']  # assuming all images are normalize to 512x512

            new_bboxes_part = []

            for each_bbox in part_bbox_pred:
                bounding_box = [int(each_bbox[0] * 512),
                                int(each_bbox[1] * 512),
                                int((each_bbox[0] + each_bbox[2]) * 512),
                                int((each_bbox[1] + each_bbox[3]) * 512),
                                ]
                new_bboxes_part.append(bounding_box)

            part_bboxes.append(new_bboxes_part)
            normalized_part_bboxes.append(part_bbox_pred)

        # visualize_bbox(image, f"{save_dir}/{img_name}.png", bboxes)
        # save filtered bbox
        bboxes = {}
        bboxes['global_normalized_bbox'] = normalized_global_bboxes
        bboxes['global_bbox'] = new_bboxes

        bboxes['part_normalized_bbox'] = normalized_part_bboxes
        bboxes['part_bbox'] = part_bboxes


        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/{img_name}.npy", bboxes)


def post_processing(label_dir, img_dir, save_dir):
    for img_path in glob.glob(img_dir+"/*"):
        img_name = os.path.basename(img_path)[:-4]

        label_path = f"{label_dir}/{img_name}.npy"
        object_info = np.load(label_path, allow_pickle=True).item()
        bbox_pred = object_info['part_normalized_bbox']  # assuming all images are normalize to 512x512
        print(img_path)

        image = PIL.Image.fromarray(cv2.imread(img_path)).resize((512, 512))
        image = np.array(image)
        new_bboxes = []

        w, h = image.shape[0], image.shape[1]
        for each_bbox in bbox_pred:
            bounding_box = [int(each_bbox[0] * w),
                            int(each_bbox[1] * h),
                            int((each_bbox[0] + each_bbox[2]) * w),
                            int((each_bbox[1] + each_bbox[3]) * h),
                            ]
            new_bboxes.append(bounding_box)
        bbox = nms(new_bboxes, iou_threshold=0.9)
        normalized_bboxes = []
        for each_original_bbox in bbox:
            part_normalized_bbox = [each_original_bbox[0] / w,
                                    each_original_bbox[1] / h,
                                    (each_original_bbox[2] - each_original_bbox[0]) / w,
                                    (each_original_bbox[3] - each_original_bbox[1]) / h]
            normalized_bboxes.append(part_normalized_bbox)



        # save filtered bbox
        bboxes = {}
        bboxes['part_normalized_bbox'] = normalized_bboxes
        bboxes['bbox'] = bbox
        np.save(f"{save_dir}/{img_name}.npy",  bboxes)
        visualize_bbox(image, f"{save_dir}/{img_name}.npy", bbox)
        
        
        


