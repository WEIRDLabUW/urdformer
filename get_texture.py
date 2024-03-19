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
from texture import load_texture, load_kitchen_texture
from utils import write_numpy
from utils import write_urdfs
from grounding_dino.detection import detector
from grounding_dino.post_processing import post_processing
import os
import time
import glob
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


def get_texture(scene):
    
    if scene=="kitchen":
        for img_path in glob.glob("images/*"):
            image_global = np.array(Image.open(img_path).convert("RGB"))

            test_name = os.path.basename(img_path)[:-4]
            detect_img = image_global.copy()
            texture_path = "texture/{0}".format(test_name)
            bbox = []
            data_path = f"grounding_dino/labels_manual/all/{test_name}.npy"
            data = np.load(data_path, allow_pickle=True).item()
            
            for boxid, each_bbox in enumerate(data['global_normalized_bbox']):
                bbox.append(each_bbox)

                bounding_box = [int(each_bbox[0] * image_global.shape[0]),
                                int(each_bbox[1] * image_global.shape[1]),
                                int((each_bbox[0] + each_bbox[2]) * image_global.shape[0]),
                                int((each_bbox[1] + each_bbox[3]) * image_global.shape[1]),
                                ]


                detect_img = cv2.rectangle(detect_img, (bounding_box[1], bounding_box[0]), (bounding_box[3], bounding_box[2]), (255, 0, 0), 6)

            for mesh_id, each_bbox in enumerate(bbox):
                # get the cropped image
                bounding_box = [int(bbox[mesh_id][0] * image_global.shape[0]),
                                int(bbox[mesh_id][1] * image_global.shape[1]),
                                int((bbox[mesh_id][0] + bbox[mesh_id][2]) * image_global.shape[0]),
                                int((bbox[mesh_id][1] + bbox[mesh_id][3]) * image_global.shape[1]),
                                ]
                cropped_image = image_global[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]

                all_parts_bbox = []
                for boxid, each_bbox in enumerate(data['part_normalized_bbox'][mesh_id]):
                    all_parts_bbox.append(each_bbox)

                bbox_parts = []
                for each_bbox_part in all_parts_bbox:
                    bounding_box_parts = [int(each_bbox_part[0] * cropped_image.shape[0]),
                                          int(each_bbox_part[1] * cropped_image.shape[1]),
                                          int((each_bbox_part[0] + each_bbox_part[2]) * cropped_image.shape[0]),
                                          int((each_bbox_part[1] + each_bbox_part[3]) * cropped_image.shape[1]),
                                          ]
                    bbox_parts.append(bounding_box_parts)

                    # part_global_bbox = [bounding_box_parts[0]+bounding_box[0],bounding_box_parts[2] - bounding_box[0], bounding_box_parts[1] + bounding_box[1],bounding_box_parts[3] - bounding_box[1]]
                    # breakpoint()
                    # detect_img = cv2.rectangle(detect_img, (part_global_bbox[1], part_global_bbox[0]), (part_global_bbox[3], part_global_bbox[2]), (255, 0, 0), 6)

                load_kitchen_texture(cropped_image, test_name, mesh_id, bbox_parts)
    else:
        for img_path in glob.glob("images/*"):
            print('image path', img_path)
            test_name = os.path.basename(img_path)[:-4]
            #############################################
            label_path = f'grounding_dino/labels_manual/{test_name}.npy'
            random=False
            texture_list = load_texture(img_path, label_path, if_random=random)
        
        
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-scene_type', '--scene_type', default='object', type=str)
    args = parser.parse_args()

    get_texture(args.scene)
