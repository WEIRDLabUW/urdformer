# Copyright (c) OpenMMLab. All rights reserved.
# modified by Zoey Chen for URDFormer

from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
import torch
import glob
import PIL
import cv2


def average_checkpoints(checkpoint_path1, checkpoint_path2, output_path):
    # Load the checkpoints
    checkpoint1 = torch.load(checkpoint_path1)
    checkpoint2 = torch.load(checkpoint_path2)

    # Get the intersection of the keys from both state dicts
    common_keys = set(checkpoint1['state_dict'].keys()) & set(checkpoint2.keys())

    # Average the weights for common keys
    averaged_state_dict = checkpoint1['state_dict'].copy()  # Start with the state dict of checkpoint1
    for key in common_keys:
        averaged_state_dict[key] = (0.5*checkpoint1['state_dict'][key] + 0.5*checkpoint2[key])

    # Create a new checkpoint dictionary with the averaged weights for common keys
    new_checkpoint = {
        'state_dict': averaged_state_dict,
        'meta': checkpoint1.get('meta', {}),  # Use meta from checkpoint1
        'message_hub': checkpoint1.get('message_hub', {}),  # Use message_hub from checkpoint1
        'optimizer': checkpoint1.get('optimizer', {}),  # Use optimizer from checkpoint1
        'param_schedulers': checkpoint1.get('param_schedulers', {})  # Use param_schedulers from checkpoint1
    }

    # Save the new checkpoint
    torch.save(new_checkpoint, output_path)
    print(f"New checkpoint with averaged weights has been saved to {output_path}")



def detector(scene_type, call_args):
    if scene_type!="kitchen":
        call_args['weights'] = 'grounding_dino/object_souped.pth'
        call_args['model'] = 'grounding_dino/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_urdformer.py'
        if scene_type == "cabinet":
            call_args['texts'] = 'drawer . cabinet_door . handle . knob . right_door . left_door'
        elif scene_type == "fridge":
            call_args['texts'] = 'drawer .  handle .  right_door . left_door . fridge_door '
        elif scene_type == "oven":
            call_args['texts'] = 'drawer .  handle . oven_door '
        elif scene_type == "dishwasher":
            call_args['texts'] = 'handle .  dishwasher_door . dishwasher_handle'
        elif scene_type == "washer":
            call_args['texts'] = 'washer . door . machine . ' \
                                 'washer_door . laundry door . circle door'
        else:
            call_args['texts'] = 'drawer . cabinet_door . handle . knob . right_door . left_door' \
                                '. washer_door . ' \
                                'dishwasher_door . oven_door . fridge_door washer . door . machine . ' \
                                'washer_door . laundry door . circle door'
    else:
        call_args['weights'] = 'grounding_dino/kitchen_souped.pth'
        call_args['model'] = 'grounding_dino/configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_urdformer.py'
        call_args['texts'] = 'object . cabinet . oven . dishwasher . fridge . oven_hood .'

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    inferencer = DetInferencer(**init_args)
    inferencer(**call_args)
