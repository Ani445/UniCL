import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from config import get_config
from datasets import voc
from model.model import build_unicl_model
import matplotlib.pyplot as plt
import yaml

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

# easy config modification
parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
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

# distributed training
parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')

def load_cls_dataset(cfg, args):
    pass
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
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATASET.NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )
    return val_loader

def test_unicl_classification(cfg, args):
    
    model = build_unicl_model(cfg, args)
    model = model.cuda()
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
        image = image.cuda()
        cls_label = cls_label.cuda()
        
        # print(image_name)
        # print(image.shape, cls_label.shape)
        # print(image, cls_label)
        
        # return
        
        with torch.no_grad():
            text_inputs = tokenizer(
                ["a photo of an cat"],
                max_length=77,         # Set the maximum length to match the model's expectation
                padding="max_length",  # Pad the sequence to the maximum length
                truncation=True,       # Truncate the sequence if it's longer than max_length
                return_tensors="pt"    # Return PyTorch tensors
            )
            image_features, text_features, T = model(image, text_inputs.to(image.device))
            logits_per_image = T * image_features @ text_features.t()
            probs = logits_per_image.softmax(dim=-1)
            print(probs)
            
    
    # print('Accuracy:', matched / total_step * cfg.dataset.crop_size * cfg.dataset.crop_size)
        
if __name__ == '__main__':
    args = parser.parse_args()
    cfg = get_config(args)
    # print(cfg.DATASET)
    # print(args)
    # print(cfg)
    test_unicl_classification(cfg, args)