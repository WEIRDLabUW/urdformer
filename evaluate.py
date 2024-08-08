import numpy as np
import PIL
from urdformer import URDFormer
import json
import torch
import cv2
import pybullet as p
from utils import visualization_parts, create_obj
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import argparse
import os
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
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        normalize,
    ])

    return preprocessing

def evaluate_full_with_masks(data_path, num_roots):
    max_bbox = 32

    data = np.load(data_path, allow_pickle=True).item()
    new_img = data['rgb']
    # new_img = cv2.cvtColor( data['rgb'], cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(new_img).resize((224, 224))

    # img_pil = PIL.Image.fromarray(data['rgb']).resize((224, 224))
    gt_position_start = data['positions_start']
    gt_position_end = data['positions_end']
    gt_relation = data['relations']
    gt_mesh = data['meshes']


    img_transform = image_transform()
    image_tensor = img_transform(img_pil)

    bbox = []
    resized_mask = []

    for maskid in range(len(data['normalized_bbox'])):
        bbox.append(data['normalized_bbox'][maskid])
        resized = np.array(Image.fromarray(data['segms'][maskid]).resize((14, 14), Image.NEAREST)) / 255
        resized_mask.append(resized)

    padded_bbox = np.zeros((max_bbox, 4))
    padded_bbox[:len(bbox)] = bbox

    padded_masks = np.zeros((max_bbox, 14, 14))
    padded_masks[:len(resized_mask)] = resized_mask

    base_type = data['base_type']

    tgt_padding_mask = torch.ones([max_bbox])
    tgt_padding_mask[:len(bbox)] = 0.0
    tgt_padding_mask = tgt_padding_mask.bool()

    tgt_padding_relation_mask = torch.ones([max_bbox + num_roots])
    tgt_padding_relation_mask[:len(bbox) + num_roots] = 0.0
    tgt_padding_relation_mask = tgt_padding_relation_mask.bool()


    return image_tensor, base_type, np.array([padded_bbox]), np.array([padded_masks]), gt_position_start, gt_position_end, gt_mesh, gt_relation, tgt_padding_mask, tgt_padding_relation_mask

def evaluate_parts_with_masks(data_path, cropped_image, num_roots, bboxid):
    max_bbox = 32

    data = np.load(data_path, allow_pickle=True).item()
    img_pil = PIL.Image.fromarray(cropped_image).resize((224, 224))
    # gt_position_start = data['part_positions_start']
    # gt_position_end = data['part_positions_end']
    # gt_relation = data['part_relations']
    # gt_mesh = data['part_meshes']

    img_transform = image_transform()
    image_tensor = img_transform(img_pil)

    bbox = []
    resized_mask = []

    for boxid, each_bbox in enumerate(data['part_normalized_bbox'][bboxid]):

        bbox.append(each_bbox)
        resized = np.zeros((14, 14))#np.array(Image.fromarray(np.array(data['part_segms'][boxid])).resize((14, 14), Image.NEAREST)) / 255
        resized_mask.append(resized)

    padded_bbox = np.zeros((max_bbox, 4))
    padded_bbox[:len(bbox)] = bbox

    padded_masks = np.zeros((max_bbox, 14, 14))
    padded_masks[:len(resized_mask)] = resized_mask

    base_type = data['base_type']

    tgt_padding_mask = torch.ones([max_bbox])
    tgt_padding_mask[:len(bbox)] = 0.0
    tgt_padding_mask = tgt_padding_mask.bool()

    tgt_padding_relation_mask = torch.ones([max_bbox + num_roots])
    tgt_padding_relation_mask[:len(bbox) + num_roots] = 0.0
    tgt_padding_relation_mask = tgt_padding_relation_mask.bool()


    return image_tensor, base_type, np.array([padded_bbox]), np.array([padded_masks]), tgt_padding_mask, tgt_padding_relation_mask

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




def process_prediction(position_pred_global, position_pred_end_global,  global_relations, part_meshes, part_positions_starts, part_positions_ends, part_relations, base_pred):
    new_relations = get_binary_relation(global_relations, position_pred_global, 5)
    new_part_relations = get_binary_relation_parts(part_relations, part_positions_starts, 1)

    pred_data = {}

    for obj_id, each_obj in enumerate(base_pred):
        if each_obj not in [1, 2, 3, 4, 5, 7]:  # if its not cabinet, shelf, oven, dishwasher, washer and fridge, count as rigid
            part_meshes[obj_id] = []
            part_positions_starts[obj_id] = []
            part_positions_ends[obj_id] = []
            new_part_relations[obj_id] = np.zeros((1, 1, 6))

    pred_data['positions_start'] = position_pred_global
    pred_data['positions_end'] = position_pred_end_global
    pred_data['relations'] = new_relations

    pred_data['part_meshes'] = part_meshes
    pred_data['part_positions_start'] = part_positions_starts
    pred_data['part_positions_end'] = part_positions_ends
    pred_data['part_relations'] = new_part_relations
    pred_data['part_bases'] = base_pred


    return pred_data

def load_kitchen_texture(asset_path, image, test_name, object_id, bboxes):
    # load the image
    # get the bounding box for drawer and doors
    texture_path = f"{asset_path}/kitchens/textures/{test_name}"

    os.makedirs(texture_path + '/{0}'.format(object_id), exist_ok=True)

    # create folder for the texture
    side_texture = "default_textures/inside.jpg"
    side_image = cv2.imread(side_texture)
    texture_list = []

    for bbox_id, each_bbox in enumerate(bboxes):
        if os.path.exists(texture_path + "/{0}/{1}.png".format(object_id,  bbox_id)):
            texture_list.append(texture_path + "/{0}/{1}.png".format(object_id, bbox_id))
            continue
        threshold1 = 0
        front_image = image[each_bbox[0]+threshold1:each_bbox[2]-threshold1, each_bbox[1]+threshold1:each_bbox[3]-threshold1]
        w,h,_ = front_image.shape

        inside_bbox = []
        for inside_id, each_bbox1 in enumerate(bboxes):
            if inside_id ==bbox_id:
                continue
            if is_inside(each_bbox1, each_bbox):
                inside_bbox.append(each_bbox1)

        # resize everything to 512x512
        inpaint_img = PIL.Image.fromarray(front_image).resize((512, 512))
        inpaint_mask = np.zeros((w,h))
        threshold = 1
        for each_inside_bbox in inside_bbox:
            inpaint_mask[max(0, each_inside_bbox[0] - each_bbox[0]-threshold):each_inside_bbox[2] - each_bbox[0]+threshold, max(0, each_inside_bbox[1] - each_bbox[1]-threshold):each_inside_bbox[3] - each_bbox[1]+threshold]=255

        inpaint_mask = PIL.Image.fromarray(cv2.resize(inpaint_mask, (512, 512), interpolation=cv2.INTER_NEAREST))

        # impaint the texture
        # inpaint_img = upscale([inpaint_img])[0].resize((512, 512))
        new_image = in_paint_pipe(prompt="panel texture, original color, smooth texture, Intricately Detailed, 16k, natural lighting, Best Quality, Masterpiece, photorealistic", image=inpaint_img, mask_image=inpaint_mask).images[0]

        if not is_small(each_bbox, 10) or len(bboxes)==1: # use the drawer color to be the base
            base_image = np.array(new_image.resize((200, 200)))
            base_texture = np.zeros((600, 600, 3))
            base_texture[200:400, :200, :] = np.rot90(base_image)
            base_texture[400:600, 400:600, :] = np.rot90(base_image)
            base_texture[200:400, 200:400, :] = np.rot90(base_image)
            base_texture[400:600, 200:400, :] = np.rot90(base_image)
            base_texture[200:400, 400:600, :] = np.rot90(base_image)
            base_texture[400:600, :200, :] = np.rot90(base_image)
            PIL.Image.fromarray(base_texture.astype(np.uint8)).save(
                texture_path + "/{0}/base.png".format(object_id))

        new_image = np.array(new_image.resize((200, 200)))
        # putting this together with side images.
        texture_map = np.zeros((600, 600, 3))
        texture_map[200:400, :200, :] = np.rot90(new_image)
        texture_map[400:600, 400:600, :] = np.rot90(new_image)
        texture_map[200:400, 200:400, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[400:600, 200:400, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[200:400, 400:600, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[400:600, :200, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        # save


        PIL.Image.fromarray(texture_map.astype(np.uint8)).save(
            texture_path + "/{0}/{1}.png".format(object_id, bbox_id))
        texture_list.append(texture_path + "/{0}/{1}.png".format(object_id, bbox_id))
    return texture_list

def get_texture(asset_path, scene, test_id):
    img_path = f"{asset_path}/{scene}/images/test{test_id}.jpg"
    image_global = np.array(Image.open(img_path).convert("RGB"))

    test_name = os.path.basename(img_path)[:-4]
    detect_img = image_global.copy()
    bbox = []
    data_path = f'assets/{scene}/labels/label{test_id}.npy'
    data = np.load(data_path, allow_pickle=True).item()

    for boxid, each_bbox in enumerate(data['normalized_bbox']):
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

        load_kitchen_texture(asset_path, cropped_image, test_name, mesh_id, bbox_parts)

def evaluate(device, data_path, asset_path, test_id, urdformer_global, urdformer_obj, if_random, texture):
    with torch.no_grad():
        #################### global scene prediction ######################
        num_roots_global = 5
        image_tensor, _, bbox, masks, position_gt, position_end_gt, mesh_gt, parent_gt, tgt_padding_mask, tgt_padding_relation_mask = evaluate_full_with_masks(data_path, num_roots_global)

        position_pred_global, position_pred_end_global, mesh_pred_global, parent_pred_global, base_pred = evaluate_real_image(image_tensor, bbox, masks, tgt_padding_mask, tgt_padding_relation_mask, urdformer_global, device)
        scale_pred_global = abs(np.array((position_pred_end_global - position_pred_global)))
        data = np.load(data_path, allow_pickle=True).item()
        image_global = cv2.cvtColor( data['rgb'], cv2.COLOR_BGR2RGB)

        front_object_position_end = []
        num_roots_part = 1


        parent_pred_parts = []
        position_pred_end_parts = []
        position_pred_start_parts = []
        mesh_pred_parts = []
        base_types = []

        for mesh_id, each_mesh in enumerate(mesh_pred_global):
            each_parent = np.unravel_index(np.argmax(parent_pred_global[num_roots_global + mesh_id]), parent_pred_global[num_roots_global + mesh_id].shape)

            parent_id = each_parent[0]

            if parent_id==4:
                continue
            # get the cropped image
            bounding_box = [int(bbox[0][mesh_id][0]*image_global.shape[0]),
                            int(bbox[0][mesh_id][1] * image_global.shape[1]),
                            int((bbox[0][mesh_id][0]+bbox[0][mesh_id][2])* image_global.shape[0]),
                            int((bbox[0][mesh_id][1]+bbox[0][mesh_id][3]) * image_global.shape[1]),
                            ]
            cropped_image = image_global[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]]
            image_tensor_part, _, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part = evaluate_parts_with_masks(data_path, cropped_image, num_roots_part, mesh_id)


            position_pred_part, position_pred_end_part, mesh_pred_part, parent_pred_part, base_pred = evaluate_real_image(
                image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part, urdformer_obj, device)

            ##################################################
            parent_pred_parts.append( np.array(parent_pred_part))
            position_pred_end_parts.append(np.array(position_pred_end_part[:, 1:]))
            position_pred_start_parts.append(np.array(position_pred_part[:, 1:]))
            mesh_pred_parts.append( np.array(mesh_pred_part))
            base_types.append(base_pred[0])
            #############################################

            if parent_id<=2:
                root_position = position_pred_global[mesh_id] + np.array([1.2, 0.2, 0])
                root_orientation = [0, 0, 0, 1]
                root_scale = scale_pred_global[mesh_id]

            elif parent_id==3:
                root_orientation = Rot.from_rotvec([0,0,np.pi/2]).as_quat()
                root_scale = [scale_pred_global[mesh_id][1], scale_pred_global[mesh_id][0], scale_pred_global[mesh_id][2]]
                root_position = position_pred_global[mesh_id] + np.array([2, 1.2, 0])

            size_scale = 4
            scale_pred_part = abs(np.array(size_scale * (position_pred_end_part - position_pred_part) / 12))
            scale_pred_part[:, 1] *= root_scale[1]
            scale_pred_part[:, 2] *= root_scale[2]
            texture_list = []
            if texture:
                '''We already saved all the texture maps for each parts, the way we get them is the same as get_texture.py, with very small modification on path and names, 
                  but in case you want to run it yourself, you can do:
                  get_texture(asset_path, scene_name, test_id) 
                '''
                label_path = f'{asset_path}/kitchens/labels/label{test_id}.npy'
                object_info = np.load(label_path, allow_pickle=True).item()
                bboxes = object_info['part_normalized_bbox'][mesh_id]

                for bbox_id in range(len(bboxes)):
                    if os.path.exists(f"{asset_path}/kitchens/textures/test{test_id}/{mesh_id}/{bbox_id}.png"):
                        texture_list.append(
                            f"{asset_path}/kitchens/textures/test{test_id}/{mesh_id}/{bbox_id}.png")
                    else:
                        print('no texture map found!')


            visualization_parts(p, root_position, root_orientation, root_scale,  base_pred[0], position_pred_part, scale_pred_part, mesh_pred_part, parent_pred_part, texture_list, if_random, filename="output")

            if parent_id<=2:
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
                image_tensor_part, _, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part = evaluate_parts_with_masks(
                    data_path, cropped_image, num_roots_part, mesh_id)

                position_pred_part, position_pred_end_part, mesh_pred_part, parent_pred_part, base_pred = evaluate_real_image(
                    image_tensor_part, bbox_part, masks_part, tgt_padding_mask_part, tgt_padding_relation_mask_part,
                    urdformer_obj, device)

                ##################################################
                parent_pred_parts.append(np.array(parent_pred_part))
                position_pred_end_parts.append(np.array(position_pred_end_part[:, 1:]))
                position_pred_start_parts.append(np.array(position_pred_part[:, 1:]))
                mesh_pred_parts.append(np.array(mesh_pred_part))
                base_types.append(base_pred[0])
                #############################################

                root_orientation = Rot.from_rotvec([0, 0, -np.pi / 2]).as_quat()
                root_scale = [1, scale_pred_global[mesh_id][0],
                              scale_pred_global[mesh_id][2]]

                root_position = position_pred_global[mesh_id] + np.array([0, 0, 0])
                root_position[1] = right_wall_distance

                size_scale = 4
                scale_pred_part = abs(np.array(size_scale * (position_pred_end_part - position_pred_part) / 12))
                scale_pred_part[:, 0] *= root_scale[0]
                scale_pred_part[:, 2] *= root_scale[2]

                texture_list = []
                if texture:
                    ############## load texture if needed ##################
                    label_path = f'{asset_path}/kitchens/labels/label{test_id}.npy'
                    object_info = np.load(label_path, allow_pickle=True).item()
                    bboxes = object_info['part_normalized_bbox'][mesh_id]

                    for bbox_id in range(len(bboxes)):
                        if os.path.exists(f"{asset_path}/kitchens/textures/test{test_id}/{mesh_id}/{bbox_id}.png"):
                            texture_list.append(f"{asset_path}/kitchens/textures/test{test_id}/{mesh_id}/{bbox_id}.png")
                        else:
                            print('no texture map found!')

                visualization_parts(p, root_position, root_orientation, root_scale,  base_pred[0], position_pred_part, scale_pred_part, mesh_pred_part, parent_pred_part, texture_list, if_random, filename="output")






def main():
    device = "cuda"
    scene_name = "kitchens"
    if_random = False
    texture = True
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(1, lightPosition=(1250, 100, 2000), rgbBackground=(1, 1, 1))
    num_relations = 6
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

    urdformer_global.eval()
    urdformer_part.eval()

    for test_id in range(54):
        p.resetSimulation()
        asset_path = "/assets"
        data_path = f"/{asset_path}/{scene_name}/labels/label{test_id}.npy" # replace it with your data path

        evaluate(device, data_path, asset_path, test_id, urdformer_global, urdformer_part, if_random, texture)

if __name__ == "__main__":
    main()
