'''
*@author: wangxinggaung
*@data: 2022-12-30
*@version: 1.0.0
'''
import os
from pyexpat import model
import torch
from LI_chaoqun.Tent_UNet.Loss import DiceLoss, entropy_loss
from LI_chaoqun.Tent_UNet.unet.Unet_model_tent import UNet_tent
import pytorch_lightning as pl
from pl_model import MyminiModel,SSLModel,SSLModel_modify
from tttdataset import UniDataset,SSLdataset,TwoStreamBatchSampler
from torch.utils.data import DataLoader

def main_2():
    train_dataset  = SSLdataset(labeled_file='',unlabeled_file='')
    val_dataset = UniDataset('')
    labeled_indices = list(range(train_dataset.num_labeled))
    unlabeled_indices = list(range(train_dataset.num_labeled, len(train_dataset)))
    bsampler = TwoStreamBatchSampler(primary_indices= unlabeled_indices,secondary_indices= labeled_indices, batch_size=16, secondary_batch_size=8)
    train_dl = DataLoader(train_dataset, batch_sampler=bsampler,num_workers=8,pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=4, shuffle=False,num_workers=8,pin_memory=True)
    net=UNet_tent(3,1)
    ema_net=UNet_tent(3,1)
    model=SSLModel_modify(net,ema_net,primary_batch_size=12)
    trainer = pl.Trainer(gpus=2,max_epochs=500,precision=16,strategy='ddp',replace_sampler_ddp=False)
    trainer.fit(model,train_dl,val_dl)
    

def main():
    train_dataset = UniDataset('tolearncode/UDA/TCIA_2.txt')
    val_dataset = UniDataset('tolearncode/UDA/TNBC_2.txt')
    train_dl = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=4, shuffle=True)
    net=UNet_tent(3,1)
    criterion=DiceLoss()
    model=MyminiModel(net,criterion)
    trainer=pl.Trainer(gpus=2,max_epochs=100,precision=16)
    trainer.fit(model,train_dl,val_dl)
    
    
    
    
if __name__=='__main__':
    main_2()