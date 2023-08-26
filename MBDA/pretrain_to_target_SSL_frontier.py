import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from yaml import parse
join = os.path.join
import numpy as np
import torch
from torch.utils.data import DataLoader
import monai
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,
    EnsureTyped,
    EnsureType,
    Resized,
    NormalizeIntensityd,
)
from dataset.dataset import TwoStreamBatchSampler
# from models.unetr2d import UNETR2D
from dataset.dataset import CustomDataset,CustomDataset_NIPS,CustomDataset_CRC
from plmodule.NIPS_SSL import MeanTeacher
from plmodule.BBDA import FrontierDistill
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from utils.load_plmodel import load_stat_dict
from datetime import datetime
import random
import shutil
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()
def _init_fn(worker_id):
    np.random.seed(int(1029)+worker_id)
parser = argparse.ArgumentParser("Baseline for Microscopy image segmentation")
# Dataset parameters
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=100.0, help='consistency_rampup')
parser.add_argument(
                    "--work_dir", default="/media/oem/ef28ddc4-7a53-4b18-a49c-3b0a5ba02fda/wxg/work_dir/work_dirs/bbda/frontier", \
                        help="path where to save models and logs"
)
# parser.add_argument("--resume", default=False, help="resume from checkpoint")
parser.add_argument("--num_workers", default=8, type=int)

# Model parameters
parser.add_argument(
                    "--model_name", default="unet", help="select mode: unet, swinunetr"
)
parser.add_argument("--num_class", default=2, type=int, help="segmentation classes")
parser.add_argument(
                    "--input_size", default=256, type=int, help="input_size"
)
# Training parameters
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU")
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument("--max_epochs", default=500, type=int)
parser.add_argument("--val_interval", default=5, type=int)
parser.add_argument("--epoch_tolerance", default=300, type=int)
parser.add_argument("--initial_lr", type=float, default=6e-4, help="learning rate")
parser.add_argument("--pretrained_model_path", type=str, default='best_model.ckpt', help="pretrained model path")
parser.add_argument("--sw_batch_size", type=str, default=None, help="sw_batch_size")
parser.add_argument("--roi_size", type=str, default=None, help="roi_size")
args = parser.parse_args()
def bbda(args):
    model_path = join(args.work_dir, "checkpoints")
    os.makedirs(model_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    shutil.copyfile(
        __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
    )
    root_dir = join(model_path, run_id + "_" + os.path.basename(__file__).split(".")[0])
    if args.model_name.lower() == "unet":
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        ema_model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=args.num_class,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        for param in ema_model.parameters():
            param.detach_()

    if args.model_name.lower() == "swinunetr":
        model = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=2,
        )

    train_transforms = Compose(
        [
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["img"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["img"], allow_missing_keys=True
            ),  # Do not scale label
            # ScaleIntensityd(keys=["label"],dtype=np.uint8, allow_missing_keys=True),#label 256->1
            SpatialPadd(keys=["img", "label"], spatial_size=args.input_size),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=args.input_size, random_size=False
            ),
            EnsureTyped(keys=["img", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=["img"], channel_dim=-1, allow_missing_keys=True),
            RandSpatialCropd(
                keys=["img", "label"], roi_size=512, random_size=False
            ),
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),

            EnsureTyped(keys=["img", "label"]),
        ]
    )
    net1 = monai.networks.nets.SwinUNETR(
            img_size=(args.input_size, args.input_size),
            in_channels=3,
            out_channels=args.num_class,
            feature_size=24,  # should be divisible by 12
            spatial_dims=2,
        )
   tr_ds = CustomDataset_CRC(labeled_file='breast_train.txt',transform=train_transforms)

    val_ds = CustomDataset_CRC(labeled_file='breast_val.txt',transform=val_transforms)
    train_loader = DataLoader(tr_ds,batch_size=16,\
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),worker_init_fn=_init_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.num_workers,pin_memory=torch.cuda.is_available())

    plmodel = FrontierDistill(args,model=model,model1=net1)

    trainer = pl.Trainer(accelerator='gpu',devices=1, max_epochs=args.max_epochs,check_val_every_n_epoch=args.val_interval,\
                auto_scale_batch_size=True,auto_select_gpus=True,default_root_dir=root_dir)#strategy=DDPStrategy(find_unused_parameters=False)
    trainer.validate(plmodel,val_loader,ckpt_path='/media/oem/ef28ddc4-7a53-4b18-a49c-3b0a5ba02fda/wxg/work_dir/work_dirs/bbda/frontier/checkpoints/20230529-1901_pretrain_to_target_SSL_frontier/lightning_logs/version_0/checkpoints/last.ckpt')
if __name__ == "__main__":
    bbda(args)