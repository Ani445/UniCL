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
import logging
from tqdm import tqdm

from model.text_encoder.build import build_tokenizer  # Add this import

MY_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

TEMPLATES = [
    '{}.',
    'a photo of a {}.',
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]


class_map = {
    0: 'aeroplane',
    1: 'bicycle',
    2: 'bird',
    3: 'boat',
    4: 'bottle',
    5: 'bus',
    6: 'car',
    7: 'cat',
    8: 'chair',
    9: 'cow',
    10: 'diningtable',
    11: 'dog',
    12: 'horse',
    13: 'motorbike',
    14: 'person',
    15: 'pottedplant',
    16: 'sheep',
    17: 'sofa',
    18: 'train',
    19: 'tvmonitor'
}


def get_text_embeddings(tokenizer, model:UniCLModel, device):
    all_embeddings = []

    for cls in MY_CLASSES:
        text_input = tokenizer(
            [
                template.format(cls) for template in TEMPLATES
            ],
            max_length=77,         # Set the maximum length to match the model's expectation
            padding="max_length",  # Pad the sequence to the maximum length
            truncation=True,       # Truncate the sequence if it's longer than max_length
            return_tensors="pt"    # Return PyTorch tensors
        )
        with torch.no_grad():
            text_features = model.encode_text(text_input.to(device))
        text_features = text_features.mean(dim=0)
        text_features /= text_features.norm()
        all_embeddings.append(text_features)

    return torch.stack(all_embeddings, dim=0)

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)
    
setup_logger()

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


torch.manual_seed(0)

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
    
    text_embeddings = get_text_embeddings(tokenizer, model, 'cuda')
    logit_scale = model.logit_scale.exp()
    
    model.eval()
    
    for i, data in enumerate(tqdm(val_loader, total=100, desc="Testing", ncols=100)):
        if i > 100:
            break
        image_name, image, cls_label = data # image_name, ori_image, image, cls_label
        # print(image_name)
        image = image.cuda()
        cls_label = cls_label.cuda()
        
        with torch.no_grad():
            image_features = model.encode_image(image)

            logits_per_image = logit_scale * image_features @ text_embeddings.t()

        # print('logits_per_image:', logits_per_image)
        
        # get the top 3 classes
        tops = torch.topk(logits_per_image, 3, dim=1)

        # Convert indices to class names
        predicted_classes = [class_map[idx] for idx in tops.indices[0].tolist()]

        # Load the image
        img = plt.imread(f'C://Users/abesh/Downloads/archive/VOC2012/JPEGImages/{image_name[0].split("/")[-1]}.jpg')
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Predicted: {predicted_classes}')
        plt.savefig(f'output/{image_name[0].split("/")[-1]}')

        # print("///////////////////////////////////////")
        # print("///////////////////////////////////////")ko
        # print('printing all_layer_image_features')
        # for idx, output in enumerate(all_layer_image_features):
        #     print(f"Output from block {idx}: {output.shape}")
    
    # print('Accuracy:', matched / total_step * cfg.dataset.crop_size * cfg.dataset.crop_size)
        
if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args)
    # print(cfg.DATASET)
    # print(args)
    # print(cfg)
    test_unicl_classification(cfg, args)


# 0: aeroplane
# 1: bicycle
# 2: bird
# 3: boat
# 4: bottle
# 5: bus
# 6: car
# 7: cat
# 8: chair
# 9: cow
# 10: diningtable
# 11: dog
# 12: horse
# 13: motorbike
# 14: person
# 15: pottedplant
# 16: sheep
# 17: sofa
# 18: train
# 19: tvmonitor
