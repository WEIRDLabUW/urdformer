import numpy as np
import PIL
from urdformer import URDFormer
import json
import torch
import cv2
import pybullet as p
from utils import visualization_global, visualization_parts, detection_config
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import argparse
from utils import write_numpy
from utils import write_urdfs
from grounding_dino.detection import detector
from grounding_dino.post_processing import post_processing, summary_kitchen
import os
import time
import glob
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
from labeller import BoundingBoxApp

# integrate the extracted texture map into URDFormer prediction
def evaluate_real_image(image_tensor, bbox, masks, tgt_padding_mask, tgt_padding_relation_mask, urdformer, device):
    rgb_input = image_tensor.float().to(device).unsqueeze(0)
    bbox_input = torch.tensor(bbox).float().to(device).unsqueeze(0)
    masks_input = torch.tensor(masks).float().to(device).unsqueeze(0)

    tgt_padding_mask = torch.logical_not(tgt_padding_mask)
    tgt_padding_mask = torch.tensor(tgt_padding_mask).to(device).unsqueeze(0)


    tgt_padding_relation_mask = torch.logical_not(tgt_padding_relation_mask)
    tgt_padding_relation_mask = torch.tensor(tgt_padding_relation_mask).to(device).unsqueeze(0)

    position_x_pred, position_y_pred, position_z_pred, position_x_end_pred, position_y_end_pred, position_z_end_pred, mesh_pred, parent_cls, base_pred = urdformer(rgb_input, bbox_input, masks_input, 2)
    position_pred_x = position_x_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_y = position_y_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_z = position_z_pred[tgt_padding_mask].argmax(dim=1)

    position_pred_x_end = position_x_end_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_y_end = position_y_end_pred[tgt_padding_mask].argmax(dim=1)
    position_pred_z_end = position_z_end_pred[tgt_padding_mask].argmax(dim=1)

    mesh_pred = mesh_pred[tgt_padding_mask].argmax(dim=1)

    base_pred = base_pred.argmax(dim=1)

    parent_pred = parent_cls[tgt_padding_relation_mask]

    position_pred = torch.stack([position_pred_x, position_pred_y, position_pred_z]).T
    position_pred_end = torch.stack([position_pred_x_end, position_pred_y_end, position_pred_z_end]).T

    return position_pred.detach().cpu().numpy(), position_pred_end.detach().cpu().numpy(), mesh_pred.detach().cpu().numpy(), parent_pred.detach().cpu().numpy(), base_pred.detach().cpu().numpy()

def image_transform():
    """Constructs the image preprocessing transform object.

    Arguments:
        image_size (int): Size of the result image
    """
    # ImageNet normalization statistics
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocessing = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    return preprocessing

def evaluate_parts_with_masks(data_path, cropped_image):
    max_bbox = 32
    num_roots = 1
    data = np.load(data_path, allow_pickle=True).item()
    img_pil = PIL.Image.fromarray(cropped_image).resize((224, 224))

    img_transform = image_transform()
    image_tensor = img_transform(img_pil)

    bbox = []
    resized_mask = []
    for boxid, each_bbox in enumerate(data['part_normalized_bbox']):
        bbox.append(each_bbox)
        resized = np.zeros((14, 14))
        resized_mask.append(resized)
    padded_bbox = np.zeros((max_bbox, 4))
    padded_bbox[:len(bbox)] = bbox

    padded_masks = np.zeros((max_bbox, 14, 14))
    padded_masks[:len(resized_mask)] = resized_mask

    tgt_padding_mask = torch.ones([max_bbox])
    tgt_padding_mask[:len(bbox)] = 0.0
    tgt_padding_mask = tgt_padding_mask.bool()

    tgt_padding_relation_mask = torch.ones([max_bbox + num_roots])
    tgt_padding_relation_mask[:len(bbox) + num_roots] = 0.0
    tgt_padding_relation_mask = tgt_padding_relation_mask.bool()

    return image_tensor, np.array([padded_bbox]), np.array([padded_masks]), tgt_padding_mask, tgt_padding_relation_mask

def get_binary_relation(global_relations, position_pred_global, num_roots):

    new_relations = np.zeros((len(position_pred_global) + num_roots, len(position_pred_global) + num_roots, 6))
    for obj_id, position in enumerate(position_pred_global):
        each_parent = np.unravel_index(np.argmax(global_relations[num_roots + obj_id]),
                                       global_relations[num_roots + obj_id].shape)
        parent_id = each_parent[0]
        relation_id = each_parent[1]
        new_relations[obj_id + num_roots, parent_id, relation_id] = 1
    return new_relations

def get_binary_relation_parts(part_relations, position_pred_part, num_roots):
    all_new_relations = []
    for obj_id, each_position in enumerate(position_pred_part):
        new_relations = np.zeros((len(position_pred_part[obj_id]) + num_roots, len(position_pred_part[obj_id]) + num_roots, 6))
        for part_id, position in enumerate(position_pred_part[obj_id]):
            part_relations[obj_id][num_roots + part_id][num_roots + part_id] = -1000000000*np.ones(6)# the parent of the one can't be itself...if so, go to the next one.
            each_parent = np.unravel_index(np.argmax(part_relations[obj_id][num_roots + part_id]), part_relations[obj_id][num_roots + part_id].shape)
            parent_id = each_parent[0]
            relation_id = each_parent[1]
            new_relations[part_id + num_roots, parent_id, relation_id] = 1
        all_new_relations.append(new_relations)
    return all_new_relations


def process_gt(gt_data):
    part_meshes = gt_data['part_meshes']
    part_positions_starts = gt_data['part_positions_start']
    part_positions_ends = gt_data['part_positions_end']
    new_part_relations = gt_data['part_relations']
    base_pred = gt_data['part_bases']

    new_starts = []
    new_ends = []
    for part_id, each_gt_start in enumerate(part_positions_starts):
        new_gt_start = [0, each_gt_start[0], each_gt_start[1]]
        new_gt_end = [0, part_positions_ends[part_id][0], part_positions_ends[part_id][1]]

        new_starts.append(new_gt_start)
        new_ends.append(new_gt_end)

    new_data = {}
    new_data['part_meshes'] = [np.array(part_meshes)]
    new_data['part_positions_start'] = [np.array(new_starts)]
    new_data['part_positions_end'] = [np.array(new_ends)]
    new_data['part_relations'] = [np.array(new_part_relations)]
    new_data['part_bases'] = [np.array(base_pred)]


    return new_data


def animate():

    time.sleep(0.2)

def process_prediction(part_meshes, part_positions_starts, part_positions_ends, part_relations, base_pred):

    new_part_relations = get_binary_relation_parts(part_relations, part_positions_starts, 1)

    pred_data = {}
    if np.array(base_pred)[0] not in [1,2,3,4,5,7]: # if its not cabinet, shelf, oven, dishwasher, washer and fridge, count as rigid
        part_meshes = []
        part_positions_starts = []
        part_positions_ends = []
        new_part_relations = np.zeros((1,1, 6))
    else:
        part_meshes = np.array(part_meshes)[0]
        part_positions_starts = np.array(part_positions_starts)[0]
        part_positions_ends = np.array(part_positions_ends)[0]
        new_part_relations = np.array(new_part_relations)[0]

    pred_data['part_meshes'] = [part_meshes]
    pred_data['part_positions_start'] = [part_positions_starts]
    pred_data['part_positions_end'] = [part_positions_ends]
    pred_data['part_relations'] = [new_part_relations]
    pred_data['part_bases'] = [np.array(base_pred)[0]]

    return pred_data


def evaluate(args, detection_args):
    input_path = args.image_path
    print('************ Applying Finetuned (Model Soup) GroundingDINO *******************')
    detector(args.scene_type, detection_args)
    # #
    # # # run postprocessing
    label_dir = 'grounding_dino/labels'
    save_dir = 'grounding_dino/labels_filtered'
    manual_dir = 'grounding_dino/labels_manual'
    post_processing(label_dir, input_path, save_dir)
    # # ask user to for manual correction
    #
    for img_path in glob.glob(f"{input_path}/*"):
        label_name = os.path.basename(img_path)[:-4]
        label_path = f"grounding_dino/labels_filtered/{label_name}.npy"
        labeled_boxes = np.load(label_path, allow_pickle=True).item()
        normalized_bboxes = labeled_boxes['part_normalized_bbox']
        root = tk.Tk()
        app = BoundingBoxApp(root, img_path, initial_boxes=normalized_bboxes, save_path = manual_dir)
        root.mainloop()

    if args.scene_type=="kitchen":
        detection_args = detection_config(args)
        # # get the part detection
        os.makedirs(f"{manual_dir}/parts/images", exist_ok=True)
        os.makedirs(f"{manual_dir}/parts/labels", exist_ok=True)
        os.makedirs(f"{manual_dir}/parts/labels_filtered", exist_ok=True)
        os.makedirs(f"{manual_dir}/parts/labels_manual", exist_ok=True)
        for each_img_path in glob.glob('images/*'):
            label_name = os.path.basename(each_img_path)[:-4]
            each_global_label = f"grounding_dino/labels_manual/{label_name}.npy"
            global_data = np.load(each_global_label, allow_pickle=True).item()
            all_bboxes = global_data['part_normalized_bbox']
            image_global = np.array(Image.open(each_img_path).convert("RGB"))
            for bbox_id, each_obj_bbox in enumerate(all_bboxes):
                bounding_box = [int(each_obj_bbox[0] * image_global.shape[0]),
                                int(each_obj_bbox[1] * image_global.shape[1]),
                                int((each_obj_bbox[0] + each_obj_bbox[2]) * image_global.shape[0]),
                                int((each_obj_bbox[1] + each_obj_bbox[3]) * image_global.shape[1]),
                                ]
                cropped_image = image_global[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]
                # save the croppd_images in to the folder and update input
                PIL.Image.fromarray(cropped_image).resize((512, 512)).save(f"{manual_dir}/parts/images/{label_name}_{bbox_id}.png")

        # run detection module again for each cropped images to get part bboxes
        detection_args['out_dir'] = "grounding_dino/labels_manual/parts/labels"
        detection_args['inputs'] = "grounding_dino/labels_manual/parts/images"
        detector('object', detection_args)
        #
        # # run postprocessing
        part_label_dir = 'grounding_dino/labels_manual/parts/labels'
        part_save_dir = 'grounding_dino/labels_manual/parts/labels_filtered'
        part_manual_dir = 'grounding_dino/labels_manual/parts/labels_manual'

        post_processing(part_label_dir, "grounding_dino/labels_manual/parts/images", part_save_dir)

        # ask user for manual correction
        for img_path in glob.glob("grounding_dino/labels_manual/parts/images/*"):

            label_name = os.path.basename(img_path)[:-4]
            label_path = f"{part_save_dir}/{label_name}.npy"
            labeled_boxes = np.load(label_path, allow_pickle=True).item()

            normalized_bboxes = labeled_boxes['part_normalized_bbox']
            root = tk.Tk()
            app = BoundingBoxApp(root, img_path, initial_boxes=normalized_bboxes, save_path = part_manual_dir)
            root.mainloop()

        save_dir = f'{manual_dir}/all'
        summary_kitchen(manual_dir, part_manual_dir, "images", save_dir)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-scene_type', '--scene_type', default='cabinet', type=str)
    parser.add_argument('-image_path', '--image_path', default='images', type=str)

    ##################### IMPORTANT! ###############################
    # URDFormer replies on good bounding boxes of parts and ojects, you can achieve this by our annotation tool (~1min label per image)
    # We also provided our finetuned GroundingDINO (model soup version) to automate this. We finetuned GroundingDino on our generated dataset, and
    # apply model soup for the pretrained and finetuned GroundingDINO. However, the perfect bbox prediction is not gauranteed and will be our future work.

    args = parser.parse_args()
    detection_args = detection_config(args) # leave the defult groundingDINO argument unchanged

    evaluate(args, detection_args)


if __name__ == "__main__":
    main()
