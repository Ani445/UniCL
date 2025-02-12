import argparse
from datasets import voc
from model.model import build_unicl_model


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/unicl_swin_tiny', help='config file path')
parser.add_argument('--unicl_model', type=str, default='pretrained/unicl_swin_tiny_patch4_window7_224.pth', help='unicl model path')

def load_dataset(cfg, args):
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='train',
        aug=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
        crop_size=cfg.dataset.crop_size,
    )
    return val_dataset


def create_val_loader(val_dataset, num_workers):
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    return val_loader

def test_unicl_classification(cfg, args):
    
    model = build_unicl_model(pretrained_path=args.unicl_model)
    model = model.cuda()
    
    val_dataset = load_dataset(cfg, args)
    val_loader = create_val_loader(val_dataset, cfg.val.num_workers)
    total_step = len(val_loader)
    matched = 0
    
    for i, (img_name, image, label) in enumerate(val_loader):
        image = image.cuda()
        label = label.cuda()
        with torch.no_grad():
            output = model(image)
            output = F.interpolate(output, size=image.size()[2:], mode='bilinear', align_corners=True)
            pred = torch.softmax(output, dim=1).argmax(0)
            matched += (pred == label)
    
    print('Accuracy:', matched / total_step * cfg.dataset.crop_size * cfg.dataset.crop_size)
        
        