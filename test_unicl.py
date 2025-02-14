import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from config import get_config
from datasets import voc
from model.model import UniCLModel, build_unicl_model
import matplotlib.pyplot as plt
import yaml
import torch.nn as nn
import logging

from model.text_encoder.build import build_tokenizer  # Add this import


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/unicl_swin_tiny.yaml', help='config file path')
parser.add_argument('--unicl_model', type=str, default='checkpoint/yfcc14m.pth', help='unicl model path')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

parser.add_argument('--batch-size', type=int, default=4, help="batch size for single GPU")
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset name')       
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--name-list-path', type=str, help='path to name list')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--output', default='output', type=str, metavar='PATH',
                    help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--debug', action='store_true', help='Perform debug only')

parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

logger = logging.getLogger(__name__)

def load_cls_dataset(cfg, args):
    val_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.DATASET.DATA_DIR,
        name_list_dir=cfg.DATASET.NAME_LIST_DIR,
        split=cfg.DATASET.SPLIT,
        stage='val',
        ignore_index=cfg.DATASET.IGNORE_INDEX,
        num_classes=cfg.DATASET.NUM_CLASSES,
        resize_shape=cfg.DATASET.IMG_SIZE,
    )
    return val_dataset


def create_val_loader(val_dataset, cfg, args):
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )
    return val_loader

all_layer_image_features = []

def hook_fn(module, input, output):
    all_layer_image_features.append(output)

def register_hooks(model:UniCLModel):
    for layer in model.image_encoder.layers:
        for block in layer.blocks:
           nn.Module.register_forward_hook(block, hook_fn)



########################################################################################
# Test UniCL Classification
########################################################################################

def test_unicl_classification(cfg, args):
    
    model = build_unicl_model(cfg, args)
    model = model.cuda()
    
    register_hooks(model)  # Register hooks to capture outputs
    
    conf_lang_encoder = cfg['MODEL']['TEXT_ENCODER']
    tokenizer = build_tokenizer(conf_lang_encoder)
    
    val_dataset = load_cls_dataset(cfg, args)
    val_loader = create_val_loader(val_dataset, cfg, args)
    total_step = len(val_loader)
    matched = 0
    
    for i, data in enumerate(val_loader):
        if i > 0:
            break
        image_name, _, image, cls_label = data # image_name, ori_image, image, cls_label
        print(image.shape)
        image = image.cuda()
        cls_label = cls_label.cuda()
        
        print("#########################")
        print("printing model parameters")
        for i, (name, param) in enumerate(model.named_parameters()):
            if i > 15:
                break
            print(i, ">>>", name, param)
        print("#########################")
        
        # print(image_name)
        # print(image.shape, cls_label.shape)
        # print(image, cls_label)
        
        # return
        
        print("*****************************")
        print("*****************************")
        print("classifying image")
        
        with torch.no_grad():
            text_inputs = tokenizer(
                ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                    'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                    ],
                max_length=77,         # Set the maximum length to match the model's expectation
                padding="max_length",  # Pad the sequence to the maximum length
                truncation=True,       # Truncate the sequence if it's longer than max_length
                return_tensors="pt"    # Return PyTorch tensors
            )
            image_features, text_features, T = model(image, text_inputs.to(image.device))
            logits_per_image = T * image_features @ text_features.t()
            print('logits_per_image:', logits_per_image)
            
            probs = torch.sigmoid(logits_per_image)
            print(probs)
        
        print("///////////////////////////////////////")
        print("///////////////////////////////////////")
        print('printing all_layer_image_features')
        for idx, output in enumerate(all_layer_image_features):
            print(f"Output from block {idx}: {output.shape}")
    
    # print('Accuracy:', matched / total_step * cfg.dataset.crop_size * cfg.dataset.crop_size)
        
if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args)
    # print(cfg.DATASET)
    # print(args)
    # print(cfg)
    test_unicl_classification(cfg, args)
