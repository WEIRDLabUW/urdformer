# remove replicated ones
import numpy as np
import cv2
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
def is_small(box, min_dim=50):
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
        if (boxA[2] - boxA[0])>450 or (boxA[3] - boxA[1])>450:

            for j, boxB in enumerate(boxes):
                if i!=j:
                    if is_inside(boxB, boxA) and is_not_too_small(boxB):
                        num_inside+=1
            if num_inside>1:
                keep_indices.discard(i)
    filtered_boxes = [boxes[i] for i in keep_indices]
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

def remove_handle_overlap(boxes, min_handle_size=50, overlap_threshold=0.1):
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
        if calculate_area(boxA) < (25 * 25):
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
    filtered_boxes = remove_biggest(filtered_boxes)
    filtered_boxes = remove_small(filtered_boxes)
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

# def remove_duplicated_handles():


def add_handle_if_needed(boxes, min_dim=50):
    # Function to determine if one box is inside another
    def is_inside(inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

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


def visualize_bbox(image, img_id, bboxes):
    for bounding_box in bboxes:


        start_point = (bounding_box[1], bounding_box[0])  # Top left corner
        end_point = (bounding_box[3], bounding_box[2])  # Bottom right corner
        color = (255, 0, 0)  # Blue color in BGR
        thickness = 3  # Line thickness

        # Draw the rectangle
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    # cv2.imshow('Image with Bounding Box', image)
    # cv2.imwrite("/home/zoeyc/github/mmdetection/outputs/kitchens/global/vis/refined/label{}.png".format(img_id), image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
#
# if the overlap is over 75%, randomly choose one to remove
# get the bounding box:

# import matplotlib.pyplot as plt
#
# # Data
# x_labels = ['pretrained', 'finetuned', 'model soup']
# y = [0.534, 0.662, 0.797]
# plt.ylim(0, 1)
# x_pos = [0, 0.4, 0.8]#range(len(x_labels))
#
# # Create a bar chart with slimmer bars and reduced gap
# plt.bar(x_pos, y, color='blue', width=0.3, align='center')
#
# # Adding numbers on top of the bars
# for i in range(len(x_pos)):
#     plt.text(x_pos[i], y[i] + 0.01, f'{y[i]:.3f}', ha='center')
#
# # Adding titles and labels
# plt.title('Performance Comparison')
# plt.xlabel('Model Type')
# plt.ylabel('Score')
#
# # Setting custom x-axis tick labels
# plt.xticks(x_pos, x_labels)
#
# # Show the plot
# plt.show()
#
# # Show the plot
# plt.show()
#
# breakpoint()

import os
from PIL import Image
import PIL
for kitchen_id in range(55):

    # object_id = 1696
    # object_info = np.load("/home/zoeyc/github/mmdetection/outputs/labels/IMG_{}.npy".format(object_id), allow_pickle=True).item()
    # bbox_pred = object_info['part_normalized_bbox'] # assuming all images are normalize to 512x512
    #
    # if os.path.isfile("/home/zoeyc/github/mmdetection/data/fridge/images/IMG_{}.jpg".format(object_id)):
    #     img_path = "/home/zoeyc/github/mmdetection/data/cabinets/images/IMG_{}.jpg".format(object_id)
    # else:
    #     img_path = "/home/zoeyc/github/mmdetection/data/cabinets/images/IMG_{}.png".format(object_id)
    object_info = np.load("/home/zoeyc/github/mmdetection/outputs/kitchens/global/labels/test{}.npy".format(kitchen_id),
                          allow_pickle=True).item()
    global_bbox_pred = object_info['part_normalized_bbox']  # assuming all images are normalize to 512x512
    img_path = "/home/zoeyc/github/mmdetection/data/kitchen/images/test{}.jpg".format(kitchen_id)


    image = cv2.imread(img_path)
    # PIL.Image.fromarray(image).show()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_bboxes = []
    for each_bbox in global_bbox_pred:
        bounding_box = [int(each_bbox[0] * 512),
                        int(each_bbox[1] * 512),
                        int((each_bbox[0] + each_bbox[2]) * 512),
                        int((each_bbox[1] + each_bbox[3]) * 512),
                        ]
        new_bboxes.append(bounding_box)
    global_bbox = nms(new_bboxes, iou_threshold=0.9)
    normalized_global_bboxes = []
    normalized_part_bboxes = []
    part_bboxes = []
    for bbox_id, each_original_bbox in enumerate(global_bbox):
        part_normalized_bbox = [each_original_bbox[0] / 512,
                                each_original_bbox[1] / 512,
                                (each_original_bbox[2] - each_original_bbox[0]) / 512,
                                (each_original_bbox[3] - each_original_bbox[1]) / 512]
        normalized_global_bboxes.append(part_normalized_bbox)




        # process part boxes
        part_info = np.load("/home/zoeyc/github/mmdetection/outputs/kitchens/parts/{0}/part{1}.npy".format(kitchen_id, bbox_id), allow_pickle=True).item()
        part_bbox_pred = part_info['part_normalized_bbox']  # assuming all images are normalize to 512x512
        img_part_path = "/home/zoeyc/github/mmdetection/data/kitchen/parts/{0}/part{1}.png".format(kitchen_id, bbox_id)
        image_part = cv2.imread(img_part_path)
        new_bboxes_part = []

        for each_bbox in part_bbox_pred:
            bounding_box = [int(each_bbox[0] * 512),
                            int(each_bbox[1] * 512),
                            int((each_bbox[0] + each_bbox[2]) * 512),
                            int((each_bbox[1] + each_bbox[3]) * 512),
                            ]
            new_bboxes_part.append(bounding_box)
        part_bbox = nms(new_bboxes_part, iou_threshold=0.9)
        part_bboxes.append(part_bbox)
        normalized_part_bboxes_per_object = []
        for bbox_id, each_original_bbox in enumerate(part_bbox):
            part_normalized_bbox = [each_original_bbox[0] / 512,
                                    each_original_bbox[1] / 512,
                                    (each_original_bbox[2] - each_original_bbox[0]) / 512,
                                    (each_original_bbox[3] - each_original_bbox[1]) / 512]
            normalized_part_bboxes_per_object.append(part_normalized_bbox)

        normalized_part_bboxes.append(normalized_part_bboxes_per_object)


    # save filtered bbox
    bboxes = {}
    bboxes['global_normalized_bbox'] = normalized_global_bboxes
    bboxes['global_bbox'] = global_bbox

    bboxes['part_normalized_bbox'] = normalized_part_bboxes
    bboxes['part_bbox'] = part_bboxes


    os.makedirs("/home/zoeyc/github/mmdetection/outputs/kitchens/all/labels_filtered", exist_ok=True)
    np.save("/home/zoeyc/github/mmdetection/outputs/kitchens/all/labels_filtered/label{}.npy".format(kitchen_id),  bboxes)


    # visualize_bbox(image, kitchen_id,bbox)

