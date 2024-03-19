import numpy as np
import PIL
from urdformer import URDFormer
import json
import torch
import cv2
import pybullet as p
from utils import visualization_global, visualization_parts, detection_config, create_obj
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import argparse
from texture import load_texture
from utils import write_numpy
from utils import write_urdfs
from grounding_dino.detection import detector
from grounding_dino.post_processing import post_processing, summary_kitchen
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

def evaluate_full_with_masks(data_path, new_img):
    max_bbox = 32
    num_roots = 5
    data = np.load(data_path, allow_pickle=True).item()
    img_pil = PIL.Image.fromarray(new_img).resize((224, 224))
    img_transform = image_transform()
    image_tensor = img_transform(img_pil)

    bbox = []
    resized_mask = []
    for boxid, each_bbox in enumerate(data['global_normalized_bbox']):
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


def evaluate_kitchen_parts_with_masks(data_path, cropped_image, bbox_id):
    max_bbox = 32
    num_roots = 1
    data = np.load(data_path, allow_pickle=True).item()
    img_pil = PIL.Image.fromarray(cropped_image).resize((224, 224))

    img_transform = image_transform()
    image_tensor = img_transform(img_pil)

    bbox = []
    resized_mask = []
    for boxid, each_bbox in enumerate(data['part_normalized_bbox'][bbox_id]):
        bbox.append(each_bbox)
        resized = np.zeros((14, 14))
        resized_mask.append(resized)

    padded_bbox = np.zeros((max_bbox, 4))
    padded_masks = np.zeros((max_bbox, 14, 14))
    tgt_padding_mask = torch.ones([max_bbox])
    tgt_padding_mask = tgt_padding_mask.bool()
    tgt_padding_relation_mask = torch.ones([max_bbox + num_roots])
    tgt_padding_relation_mask = tgt_padding_relation_mask.bool()

    if len(bbox)>0:
        padded_bbox[:len(bbox)] = bbox
        padded_masks[:len(resized_mask)] = resized_mask
        tgt_padding_mask[:len(bbox)] = 0.0
        tgt_padding_relation_mask[:len(bbox) + num_roots] = 0.0


    return image_tensor, np.array([padded_bbox]), np.array([padded_masks]), tgt_padding_mask, tgt_padding_relation_mask

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



def get_kitchen_image():
    p1 = 9  # 8
    p2 = 4  # 4.2
    c2 = 4.5  # 4.2

    p3 = 2.8
    c3 = 2.8
    view_matrix = p.computeViewMatrix([p1, p2, p3], [0, c2, c3], [0, 0, 1])
    rgb = traj_camera(view_matrix)
    return rgb



def get_camera_parameters_move(traj_i):
    all_p2s = np.arange(-0.5, 1.5, 0.1)
    p1 = 1.5
    p2 = all_p2s[traj_i]
    c2 = 0.5

    p3 = 1.2
    c3 = 0.5


    view_matrix = p.computeViewMatrix([p1, p2, p3], [0, c2, c3], [0, 0, 1])

    return view_matrix



def traj_camera(view_matrix):
    zfar, znear = 0.01, 10
    fov, aspect, nearplane, farplane = 60, 1, 0.01, 100
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, nearplane, farplane)
    light_pos = [3, 1.5, 5]
    _, _, color, depth, segm= p.getCameraImage(512, 512, view_matrix, projection_matrix, light_pos, shadow=1, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.array(color)[:,:, :3]
    return rgb

def animate(object_id, link_orientations):
    # os.makedirs(save_path, exist_ok=True)
    for i in range(20):
        for jid in range(p.getNumJoints(object_id)):
            ji = p.getJointInfo(object_id, jid)
            if ji[16]==-1 and ji[2] == 1:
                jointpos = np.random.uniform(0.2, 0.4)
                p.resetJointState(object_id, jid, jointpos)
            if ji[16]==-1 and ji[2] == 0:
                if link_orientations[int(ji[1][5:])-1][-1] == -1:
                    jointpos = np.random.uniform(0.5, 1)
                elif ji[13][1] == 1:
                    jointpos = np.random.uniform(0.25, 0.7)
                else:
                    jointpos = np.random.uniform(-0.7, -0.25)
                p.resetJointState(object_id, jid, jointpos)
        # if if_save:
        #     view_matrix = get_camera_parameters_move(i)
        #     rgb = traj_camera(view_matrix)
        #
        #     PIL.Image.fromarray(rgb).save(f"{save_path}/{i}.png")

        time.sleep(0.5)
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



def kitchen_prediction(img_path, global_label_path, urdformer_global, urdformer_obj, device, with_texture, if_random):
    gt_info = {}
    pred_info = {}
    all_link_orientations = []
    p.resetSimulation()
    #################### global scene prediction ######################
    num_roots_global = 5
    image_global = np.array(Image.open(img_path).convert("RGB"))

    #############################################################################################################################
    test_name = os.path.basename(img_path)[:-4]

    image_tensor, bbox, masks, tgt_padding_mask, tgt_padding_relation_mask = evaluate_full_with_masks(global_label_path, image_global)

    position_pred_global, position_pred_end_global, mesh_pred_global, parent_pred_global, base_pred = evaluate_real_image(
        image_tensor, bbox, masks, tgt_padding_mask, tgt_padding_relation_mask, urdformer_global, device)

    scale_pred_global = abs(np.array((position_pred_end_global - position_pred_global)))
    # visualization(p, mesh_pred_global, position_pred_global, position_pred_end_global, scale_pred_global, parent_pred_global)
    ##################### object level prediction ######################
    # get the corresponding original image:
    front_object_position_end = []
    pred_info['global_starts_pred'] = position_pred_global
    pred_info['global_ends_pred'] = position_pred_end_global

    global_parents_pred = []
    for mesh_id, each_mesh in enumerate(mesh_pred_global):
        each_parent = np.unravel_index(np.argmax(parent_pred_global[num_roots_global + mesh_id]),
                                       parent_pred_global[num_roots_global + mesh_id].shape)
        parent_id = each_parent[0]
        global_parents_pred.append(parent_id)

    pred_info['global_parents_pred'] = global_parents_pred
    pred_info['global_meshes_pred'] = [None] * len(mesh_pred_global)
    pred_info['part_starts_pred'] = [None] * len(mesh_pred_global)
    pred_info['part_ends_pred'] = [None] * len(mesh_pred_global)
    pred_info['part_parents_pred'] = [None] * len(mesh_pred_global)
    pred_info['part_meshes_pred'] = [None] * len(mesh_pred_global)

    texture_path = "textures/{0}".format(test_name)
    for mesh_id, each_mesh in enumerate(mesh_pred_global):
        each_parent = np.unravel_index(np.argmax(parent_pred_global[num_roots_global + mesh_id]),
                                       parent_pred_global[num_roots_global + mesh_id].shape)
        parent_id = each_parent[0]
        if parent_id == 4:
            continue
        # get the cropped image
        bounding_box = [int(bbox[0][mesh_id][0] * image_global.shape[0]),
                        int(bbox[0][mesh_id][1] * image_global.shape[1]),
                        int((bbox[0][mesh_id][0] + bbox[0][mesh_id][2]) * image_global.shape[0]),
                        int((bbox[0][mesh_id][1] + bbox[0][mesh_id][3]) * image_global.shape[1]),
                        ]
        cropped_image = image_global[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]
        image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part = evaluate_kitchen_parts_with_masks(
            global_label_path, cropped_image, mesh_id)

        # get texture list for each part
        bbox_part_new = bbox_part[0][torch.logical_not(tgt_padding_mask_part).numpy()]
        bbox_parts = []
        for each_bbox_part in bbox_part_new:
            bounding_box_parts = [int(each_bbox_part[0] * cropped_image.shape[0]),
                                  int(each_bbox_part[1] * cropped_image.shape[1]),
                                  int((each_bbox_part[0] + each_bbox_part[2]) * cropped_image.shape[0]),
                                  int((each_bbox_part[1] + each_bbox_part[3]) * cropped_image.shape[1]),
                                  ]
            bbox_parts.append(bounding_box_parts)

        texture_list = []
        if with_texture:
            for bbox_id, each_bbox in enumerate(bbox_parts):
                if os.path.exists(texture_path + "/{0}/{1}.png".format(mesh_id, bbox_id)):
                    texture_list.append(texture_path + "/{0}/{1}.png".format(mesh_id, bbox_id))



        position_pred_part, position_pred_end_part, mesh_pred_part, parent_pred_part, base_pred = evaluate_real_image(
            image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part,
            urdformer_obj, device)

        ##################################################
        pred_info['part_parents_pred'][mesh_id] = np.array(parent_pred_part)
        pred_info['part_starts_pred'][mesh_id] = np.array(position_pred_part)[:, 1:]
        pred_info['part_ends_pred'][mesh_id] = np.array(position_pred_end_part)[:, 1:]
        pred_info['part_meshes_pred'][mesh_id] = np.array(mesh_pred_part)
        pred_info['global_meshes_pred'][mesh_id] = base_pred[0]
        #############################################

        if parent_id <= 2:
            root_position = position_pred_global[mesh_id] + np.array([1.2, 0.2, 0])
            root_orientation = [0, 0, 0, 1]
            root_scale = scale_pred_global[mesh_id]

        elif parent_id == 3:
            root_orientation = Rot.from_rotvec([0, 0, np.pi / 2]).as_quat()
            root_scale = [scale_pred_global[mesh_id][1], scale_pred_global[mesh_id][0], scale_pred_global[mesh_id][2]]
            root_position = position_pred_global[mesh_id] + np.array([2, 1.2, 0])
        root_scale = np.array(root_scale).astype(float)
        root_scale[1] = 0.95 * root_scale[1]
        size_scale = 4

        scale_pred_part = abs(np.array(size_scale * (position_pred_end_part - position_pred_part) / 12))
        scale_pred_part[:, 1] *= root_scale[1]
        scale_pred_part[:, 2] *= root_scale[2]

        object_id, link_orientations = visualization_parts(p, root_position, root_orientation, root_scale, base_pred[0], position_pred_part,
                            scale_pred_part, mesh_pred_part, parent_pred_part, texture_list, if_random, filename=f"output/{test_name}_{mesh_id}")

        all_link_orientations.append(link_orientations)

        if parent_id <= 2:
            front_object_position_end.append(position_pred_end_global[mesh_id][1])
    for mesh_id, each_mesh in enumerate(mesh_pred_global):
        each_parent = np.unravel_index(np.argmax(parent_pred_global[num_roots_global + mesh_id]),
                                       parent_pred_global[num_roots_global + mesh_id].shape)

        parent_id = each_parent[0]
        if parent_id == 4:
            right_wall_distance = max(front_object_position_end)
            # get the cropped image
            bounding_box = [int(bbox[0][mesh_id][0] * image_global.shape[0]),
                            int(bbox[0][mesh_id][1] * image_global.shape[1]),
                            int((bbox[0][mesh_id][0] + bbox[0][mesh_id][2]) * image_global.shape[0]),
                            int((bbox[0][mesh_id][1] + bbox[0][mesh_id][3]) * image_global.shape[1]),
                            ]
            cropped_image = image_global[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]

            image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part = evaluate_kitchen_parts_with_masks(
                global_label_path, cropped_image, mesh_id)

            # get texture list for each part
            bbox_part_new = bbox_part[0][torch.logical_not(tgt_padding_mask_part).numpy()]
            bbox_parts = []
            for each_bbox_part in bbox_part_new:
                bounding_box_parts = [int(each_bbox_part[0] * cropped_image.shape[0]),
                                      int(each_bbox_part[1] * cropped_image.shape[1]),
                                      int((each_bbox_part[0] + each_bbox_part[2]) * cropped_image.shape[0]),
                                      int((each_bbox_part[1] + each_bbox_part[3]) * cropped_image.shape[1]),
                                      ]
                bbox_parts.append(bounding_box_parts)

            texture_list = []
            if with_texture:
                for bbox_id, each_bbox in enumerate(bbox_parts):
                    if os.path.exists(texture_path + "/{0}/{1}.png".format(mesh_id, bbox_id)):
                        texture_list.append(texture_path + "/{0}/{1}.png".format(mesh_id, bbox_id))

            position_pred_part, position_pred_end_part, mesh_pred_part, parent_pred_part, base_pred = evaluate_real_image(
                image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part,
                urdformer_obj, device)

            pred_info['part_starts_pred'][mesh_id] = np.array(position_pred_part)[:, 1:]
            pred_info['part_ends_pred'][mesh_id] = np.array(position_pred_end_part)[:, 1:]
            pred_info['part_parents_pred'][mesh_id] = np.array(parent_pred_part)

            pred_info['part_meshes_pred'][mesh_id] = np.array(mesh_pred_part)
            # base_types.append(base_pred[0])
            pred_info['global_meshes_pred'][mesh_id] = base_pred[0]
            #############################################

            root_orientation = Rot.from_rotvec([0, 0, -np.pi / 2]).as_quat()
            root_scale = [1, scale_pred_global[mesh_id][0],
                          scale_pred_global[mesh_id][2]]
            root_scale = np.array(root_scale).astype(float)
            root_scale[1] = 0.95 * root_scale[1]

            root_position = position_pred_global[mesh_id] + np.array([0, 0, 0])
            root_position[1] = right_wall_distance

            size_scale = 4
            scale_pred_part = abs(np.array(size_scale * (position_pred_end_part - position_pred_part) / 12))
            scale_pred_part[:, 0] *= root_scale[0]
            scale_pred_part[:, 2] *= root_scale[2]

            object_id, link_orientations = visualization_parts(p, root_position, root_orientation, root_scale, base_pred[0], position_pred_part,
                                scale_pred_part, mesh_pred_part, parent_pred_part, texture_list, if_random, filename=f"output/{test_name}_{mesh_id}")
            all_link_orientations.append(link_orientations)
    base_path = "meshes/layout"
    root_paths = ["floor", "ceiling", "front_wall", "left_wall", "right_wall"]
    for root in root_paths:
        position = [0, 0, 0]
        orientation = [0, 0, 0, 1]
        if root == "right_wall":
            position = [0, max(front_object_position_end)+1.2, 0]
        layout = create_obj(p, base_path + "/" + str(root) + ".obj", [1, 1, 1], position, orientation)
        # p.changeVisualShape(layout, -1, rgbaColor=(
        # np.random.uniform(0.7, 0.8), np.random.uniform(0.7, 0.8), np.random.uniform(0.7, 0.8), 1))
        base_texture = "default_textures/ceiling_texture/texture.png"
        base_tex = p.loadTexture(base_texture)
        p.changeVisualShape(layout, -1, rgbaColor=(1, 1, 1, 1), textureUniqueId=base_tex)

    objs = p.getNumBodies()
    # os.makedirs(f"{save_name}", exist_ok=True)

    for i in range(20):
        for obj in range(objs - 5):
            for jid in range(p.getNumJoints(obj)):
                ji = p.getJointInfo(obj, jid)
                if ji[16] == -1 and ji[2] == 1:
                    jointpos = np.random.uniform(0.2, 0.4)
                    p.resetJointState(obj, jid, jointpos)
                if ji[16] == -1 and ji[2] == 0:
                    if all_link_orientations[obj][int(ji[1][5:]) - 1][-1] == -1:
                        jointpos = np.random.uniform(0.5, 1)  # np.random.uniform(0.25, 0.45)
                    elif ji[13][1] == 1:
                        jointpos = np.random.uniform(0.25, 0.7)  # np.random.uniform(-0.5, 0.7)
                    else:
                        jointpos = np.random.uniform(-0.7, -0.25)
                    p.resetJointState(obj, jid, jointpos)
            time.sleep(0.2)
        # rgb = get_kitchen_image()
        # PIL.Image.fromarray(rgb).save(f"{save_name}/{i}.png")
    print("press enter to quit")
    input() # make a pause



def object_prediction(img_path, label_final_dir, urdformer_part, device, with_texture, if_random
                      ):

    parent_pred_parts = []
    position_pred_end_parts = []
    position_pred_start_parts = []
    mesh_pred_parts = []
    base_types = []

    test_name = os.path.basename(img_path)[:-4]
    image = np.array(PIL.Image.open(img_path).convert("RGB"))
    image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part = evaluate_parts_with_masks(
        f"{label_final_dir}/{test_name}.npy", image)

    position_pred_part, position_pred_end_part, mesh_pred_part, parent_pred_part, base_pred = evaluate_real_image(
        image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part,
        urdformer_part, device)

    size_scale = 4
    scale_pred_part = abs(np.array(size_scale * (position_pred_end_part - position_pred_part) / 12))

    root_position = [0, 0, 0]
    root_orientation = [0, 0, 0, 1]
    root_scale = [1, 1, 1]
    if base_pred[0] == 5:
        root_scale[2]*=2

    scale_pred_part[:, 2] *= root_scale[2]

    ##################################################
    parent_pred_parts.append(np.array(parent_pred_part))
    position_pred_end_parts.append(np.array(position_pred_end_part[:, 1:]))
    position_pred_start_parts.append(np.array(position_pred_part[:, 1:]))
    mesh_pred_parts.append(np.array(mesh_pred_part))
    base_types.append(base_pred[0])

    # visualization
    texture_list = []
    if with_texture:
        ############## load texture if needed ##################
        label_path = f"{label_final_dir}/{test_name}.npy"
        object_info = np.load(label_path, allow_pickle=True).item()
        bboxes = object_info['part_normalized_bbox']


        for bbox_id in range(len(bboxes)):
            if os.path.exists(f"textures/{test_name}/{bbox_id}.png"):
                texture_list.append(f"textures/{test_name}/{bbox_id}.png")
            else:
                print('no texture map found! Run get_texture.py first')


    object_id, link_orientations = visualization_parts(p, root_position, root_orientation, root_scale, base_pred[0],
                                                       position_pred_part, scale_pred_part, mesh_pred_part,
                                                       parent_pred_part, texture_list, if_random, filename=f"output/{test_name}")



    animate(object_id, link_orientations)

    root = "meshes/cabinet.obj"

    time.sleep(1)

def evaluate(args, with_texture=False):
    device = "cuda"
    input_path = args.image_path
    label_dir = "grounding_dino/labels_manual"
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(1, lightPosition=(1250, 100, 2000), rgbBackground=(1, 1, 1))

    ########################  URDFormer Core  ##############################
    # load the lobal checkpoint
    num_relations = 6
    if args.scene_type == 'kitchen':
        urdformer_global = URDFormer(num_relations=num_relations, num_roots=5)
        urdformer_global = urdformer_global.to(device)
        global_checkpoint = "checkpoints/global.pth"
        checkpoint = torch.load(global_checkpoint)
        urdformer_global.load_state_dict(checkpoint['model_state_dict'])

    urdformer_part = URDFormer(num_relations=num_relations, num_roots=1)
    urdformer_part = urdformer_part.to(device)
    part_checkpoint = "checkpoints/part.pth"
    checkpoint = torch.load(part_checkpoint)
    urdformer_part.load_state_dict(checkpoint['model_state_dict'])

    for img_path in glob.glob(input_path+"/*"):
        p.resetSimulation()
        test_name = os.path.basename(img_path)[:-4]
        # save_name = f"/home/zoeyc/github/urdformer_release/release_media/kitchen_animation/{test_name}"
        # if os.path.exists(save_name):
        #     continue
        if args.scene_type=="kitchen":
            kitchen_prediction(img_path, label_dir+f"/all/{test_name}.npy", urdformer_global, urdformer_part, device, with_texture, args.random)
        else:
            object_prediction(img_path, label_dir, urdformer_part, device, with_texture, args.random)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--texture', action='store_true', help='adding texture')
    parser.add_argument('--scene_type', '--scene_type', default='cabinet', type=str)
    parser.add_argument('--image_path', '--image_path', default='images', type=str)
    parser.add_argument('--random', '--random', action='store_true', help='use random meshes from partnet?')

    ##################### IMPORTANT! ###############################
    # URDFormer replies on good bounding boxes of parts and ojects, you can achieve this by our annotation tool (~1min label per image)
    # We also provided our finetuned GroundingDINO (model soup version) to automate/initialize this. We finetuned GroundingDino on our generated dataset, and
    # apply model soup for the pretrained and finetuned GroundingDINO. However, the performance of bbox prediction is not gauranteed and will be our future work.

    args = parser.parse_args()
    evaluate(args, with_texture=args.texture)


if __name__ == "__main__":
    main()
