'''
*@author: wangxinggaung
*@data: 2022-12-30
*@version: 1.0.0
'''
import pickle
import torch
import os
join = os.path.join
import numpy as np
import torch
from .BBDA import Basemodel
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric,HausdorffDistanceMetric,SurfaceDistanceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.transforms.post.array import VoteEnsemble
from monai.losses import DiceCELoss,DiceLoss
from monai.networks import one_hot
import pytorch_lightning as pl
from datetime import datetime
import shutil
from utils import ramps
from losses import softmax_kl_loss,Entropy
from torch.nn.functional import softmax,log_softmax,kl_div
from skimage import io,measure,morphology
from .DKD import dkd_loss
# from segmentation_models_pytorch.utils.metrics import IoU #add IOU metric
from dataset.build import RegionSelection_BBDA
from torch.nn import CrossEntropyLoss
import time
post_pred = Compose(
    [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
)
post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])

class Semi_Distill_DA_full(Basemodel):
    def __init__(self,args,model,ema_model,model1,model2,model3):
        super().__init__(args,model)
        self.ema_model = ema_model
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.contrast_loss = CrossEntropyLoss(ignore_index=-1)
    def training_step(self,batch,batch_idx):
        img,label = batch['img'],batch['label']
        ema_input = img[self.args.labeled_bs:]#无标签数据一致性计算
        noise = torch.clamp(torch.randn_like(
            ema_input) * 0.1, -0.2, 0.2)
        ema_input = ema_input + noise
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval(),self.ema_model.eval()
        with torch.no_grad():
            output_ema = self.ema_model(ema_input)
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)
            wt1,wt2,wt3 = self.cal_src_wt(output1,output2,output3)
        output_m = wt1*output1+wt2*output2+wt3*output3
###添加ensemble label####
        ensembel_label = vote_ensemble(output1,output2,output3)
        ensembel_loss = self.contrast_loss(output,ensembel_label)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        supervised_loss = self.loss(output[:self.args.labeled_bs],label[:self.args.labeled_bs])
        output_ema_soft = softmax(output_ema,dim=1)
        output_soft = softmax(output,dim=1)
        if self.global_step<50:
            consistency_loss = torch.tensor(0.0)
            consistency_loss = consistency_loss.to(self.device)
            entropy_loss = torch.tensor(0.0)
            entropy_loss = entropy_loss.to(self.device)
        else:
            consistency_loss = torch.mean(
                (output_soft[self.args.labeled_bs:]-output_ema_soft)**2)
            entropy_loss = torch.mean(Entropy(output_soft))
            msoftmax = output_soft.mean(dim=0)
            gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss -= gentropy_loss
        consistency_weight = self.get_current_consistency_weight(self.global_step)
        unsupervised_loss = consistency_weight * consistency_loss
        all_loss = supervised_loss + unsupervised_loss + distill_loss/100000 + entropy_loss/(img.shape[0]*100) + ensembel_loss
        self.update_ema_variables(self.model, self.ema_model, self.args.ema_decay, self.global_step)
        return {'loss':all_loss,'supervised_loss':supervised_loss,'unsupervised_loss':unsupervised_loss,'distill_loss':distill_loss,'entropy_loss':entropy_loss}
    def training_epoch_end(self, outputs):
        all_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()
        supervised_loss_epoch = torch.stack([x['supervised_loss'] for x in outputs]).mean()
        unsupervised_loss_epoch = torch.stack([x['unsupervised_loss'] for x in outputs]).mean()
        distill_loss_epoch = torch.stack([x['distill_loss'] for x in outputs]).mean()
        entropy_loss_epoch = torch.stack([x['entropy_loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',all_loss_epoch,prog_bar=True)
        self.log_dict({'train_supervised_loss_epoch':supervised_loss_epoch,\
                        'train_unsupervised_loss_epoch':unsupervised_loss_epoch,\
                        'train_distill_loss_epoch':distill_loss_epoch,\
                        'train_entropy_loss_epoch':entropy_loss_epoch})
        
    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            self.start_time = time.time()
        else:
            time_per_epoch = (time.time()-self.start_time)
            self.start_time = time.time()
            print(f'epoch {self.current_epoch} time:{time_per_epoch}')
    def test_step(self, batch, batch_idx):
        img,label,name = batch['img'],batch['label'],batch['idx']
        output = self.model(img)
        #save predict result as image
        output = torch.softmax(output,dim=1)
        output = torch.argmax(output,dim=1)
        mask = output[0,:,:].cpu().numpy()
        mask = mask.astype(np.uint8)
        mask = mask*255
        name = name[0]
        isave = name+'.png'
        pth = os.path.join('/media/oem/ef28ddc4-7a53-4b18-a49c-3b0a5ba02fda/wxg/work_dir/work_dirs/KIRC',isave)
        io.imsave(pth,mask)
        
        
    def get_current_consistency_weight(self,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)
    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def cal_src_wt(self,logits1,logits2,logits3):
        score1,score2,score3 = 0,0,0
        for i in range(len(logits1)):
            o1,o2,o3 = logits1[i:i + 1, :, :, :],logits2[i:i + 1, :, :, :],logits3[i:i + 1, :, :, :]
            score1,score2,score3 = RegionSelection_BBDA(o1)+score1, RegionSelection_BBDA(o2)+score2,RegionSelection_BBDA(o3)+score3
        wt1,wt2,wt3 = score1/(score1+score2+score3),score2/(score1+score2+score3),score3/(score1+score2+score3)
        return wt1,wt2,wt3
    
# TODO:class Semi_Distill_DA_ablation(pl.LightningModule):

class Semi_Distill_DA_None(Basemodel):
    def __init__(self,args,model,ema_model,model1,model2,model3):
        super().__init__(args,model)
        self.ema_model = ema_model
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.contrast_loss = CrossEntropyLoss(ignore_index=-1)
    def training_step(self,batch,batch_idx):
        img,label = batch['img'],batch['label']
        ema_input = img[self.args.labeled_bs:]#无标签数据一致性计算
        noise = torch.clamp(torch.randn_like(
            ema_input) * 0.1, -0.2, 0.2)
        ema_input = ema_input + noise
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval(),self.ema_model.eval()
        with torch.no_grad():
            output_ema = self.ema_model(ema_input)
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)
            wt1,wt2,wt3 = self.cal_src_wt(output1,output2,output3)
        # output_m = wt1*output1+wt2*output2+wt3*output3
        output_m = (output1+output2+output3)/3
###添加ensemble label####
        ensembel_label = vote_ensemble(output1,output2,output3)
        ensembel_loss = self.contrast_loss(output,ensembel_label)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        supervised_loss = self.loss(output[:self.args.labeled_bs],label[:self.args.labeled_bs])
        output_ema_soft = softmax(output_ema,dim=1)
        output_soft = softmax(output,dim=1)
        if self.global_step<50:
            consistency_loss = torch.tensor(0.0)
            consistency_loss = consistency_loss.to(self.device)
            entropy_loss = torch.tensor(0.0)
            entropy_loss = entropy_loss.to(self.device)
        else:
            consistency_loss = torch.mean(
                (output_soft[self.args.labeled_bs:]-output_ema_soft)**2)
            entropy_loss = torch.mean(Entropy(output_soft))
            msoftmax = output_soft.mean(dim=0)
            gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss -= gentropy_loss
        consistency_weight = self.get_current_consistency_weight(self.global_step)
        unsupervised_loss = consistency_weight * consistency_loss
        all_loss = supervised_loss + unsupervised_loss + distill_loss/100000 + 0*entropy_loss/(img.shape[0]*100) + 0*ensembel_loss
        self.update_ema_variables(self.model, self.ema_model, self.args.ema_decay, self.global_step)
        return {'loss':all_loss,'supervised_loss':supervised_loss,'unsupervised_loss':unsupervised_loss,'distill_loss':distill_loss,'entropy_loss':entropy_loss}
    def training_epoch_end(self, outputs):
        all_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()
        supervised_loss_epoch = torch.stack([x['supervised_loss'] for x in outputs]).mean()
        unsupervised_loss_epoch = torch.stack([x['unsupervised_loss'] for x in outputs]).mean()
        distill_loss_epoch = torch.stack([x['distill_loss'] for x in outputs]).mean()
        entropy_loss_epoch = torch.stack([x['entropy_loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',all_loss_epoch,prog_bar=True)
        self.log_dict({'train_supervised_loss_epoch':supervised_loss_epoch,\
                        'train_unsupervised_loss_epoch':unsupervised_loss_epoch,\
                        'train_distill_loss_epoch':distill_loss_epoch,\
                        'train_entropy_loss_epoch':entropy_loss_epoch})
        
    def get_current_consistency_weight(self,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)
    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    def cal_src_wt(self,logits1,logits2,logits3):
        score1,score2,score3 = 0,0,0
        for i in range(len(logits1)):
            o1,o2,o3 = logits1[i:i + 1, :, :, :],logits2[i:i + 1, :, :, :],logits3[i:i + 1, :, :, :]
            score1,score2,score3 = RegionSelection_BBDA(o1)+score1, RegionSelection_BBDA(o2)+score2,RegionSelection_BBDA(o3)+score3
        wt1,wt2,wt3 = score1/(score1+score2+score3),score2/(score1+score2+score3),score3/(score1+score2+score3)
        return wt1,wt2,wt3

class Semi_Distill_DA_Weight(Basemodel):
    def __init__(self,args,model,ema_model,model1,model2,model3):
        super().__init__(args,model)
        self.ema_model = ema_model
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.contrast_loss = CrossEntropyLoss(ignore_index=-1)
    def training_step(self,batch,batch_idx):
        img,label = batch['img'],batch['label']
        ema_input = img[self.args.labeled_bs:]#无标签数据一致性计算
        noise = torch.clamp(torch.randn_like(
            ema_input) * 0.1, -0.2, 0.2)
        ema_input = ema_input + noise
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval(),self.ema_model.eval()
        with torch.no_grad():
            output_ema = self.ema_model(ema_input)
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)
            wt1,wt2,wt3 = self.cal_src_wt(output1,output2,output3)
        output_m = wt1*output1+wt2*output2+wt3*output3
###添加ensemble label####
        ensembel_label = vote_ensemble(output1,output2,output3)
        ensembel_loss = self.contrast_loss(output,ensembel_label)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        supervised_loss = self.loss(output[:self.args.labeled_bs],label[:self.args.labeled_bs])
        output_ema_soft = softmax(output_ema,dim=1)
        output_soft = softmax(output,dim=1)
        if self.global_step<50:
            consistency_loss = torch.tensor(0.0)
            consistency_loss = consistency_loss.to(self.device)
            entropy_loss = torch.tensor(0.0)
            entropy_loss = entropy_loss.to(self.device)
        else:
            consistency_loss = torch.mean(
                (output_soft[self.args.labeled_bs:]-output_ema_soft)**2)
            entropy_loss = torch.mean(Entropy(output_soft))
            msoftmax = output_soft.mean(dim=0)
            gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss -= gentropy_loss
        consistency_weight = self.get_current_consistency_weight(self.global_step)
        unsupervised_loss = consistency_weight * consistency_loss
        all_loss = supervised_loss + unsupervised_loss + distill_loss/100000 + 0*entropy_loss/(img.shape[0]*100) + 0*ensembel_loss
        self.update_ema_variables(self.model, self.ema_model, self.args.ema_decay, self.global_step)
        return {'loss':all_loss,'supervised_loss':supervised_loss,'unsupervised_loss':unsupervised_loss,'distill_loss':distill_loss,'entropy_loss':entropy_loss}
    def training_epoch_end(self, outputs):
        all_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()
        supervised_loss_epoch = torch.stack([x['supervised_loss'] for x in outputs]).mean()
        unsupervised_loss_epoch = torch.stack([x['unsupervised_loss'] for x in outputs]).mean()
        distill_loss_epoch = torch.stack([x['distill_loss'] for x in outputs]).mean()
        entropy_loss_epoch = torch.stack([x['entropy_loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',all_loss_epoch,prog_bar=True)
        self.log_dict({'train_supervised_loss_epoch':supervised_loss_epoch,\
                        'train_unsupervised_loss_epoch':unsupervised_loss_epoch,\
                        'train_distill_loss_epoch':distill_loss_epoch,\
                        'train_entropy_loss_epoch':entropy_loss_epoch})
        
    def get_current_consistency_weight(self,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)
    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    def cal_src_wt(self,logits1,logits2,logits3):
        score1,score2,score3 = 0,0,0
        for i in range(len(logits1)):
            o1,o2,o3 = logits1[i:i + 1, :, :, :],logits2[i:i + 1, :, :, :],logits3[i:i + 1, :, :, :]
            score1,score2,score3 = RegionSelection_BBDA(o1)+score1, RegionSelection_BBDA(o2)+score2,RegionSelection_BBDA(o3)+score3
        wt1,wt2,wt3 = score1/(score1+score2+score3),score2/(score1+score2+score3),score3/(score1+score2+score3)
        return wt1,wt2,wt3
    
class Semi_Distill_DA_PL(Basemodel):
    def __init__(self,args,model,ema_model,model1,model2,model3):
        super().__init__(args,model)
        self.ema_model = ema_model
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.contrast_loss = CrossEntropyLoss(ignore_index=-1)
    def training_step(self,batch,batch_idx):
        img,label = batch['img'],batch['label']
        ema_input = img[self.args.labeled_bs:]#无标签数据一致性计算
        noise = torch.clamp(torch.randn_like(
            ema_input) * 0.1, -0.2, 0.2)
        ema_input = ema_input + noise
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval(),self.ema_model.eval()
        with torch.no_grad():
            output_ema = self.ema_model(ema_input)
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)
            wt1,wt2,wt3 = self.cal_src_wt(output1,output2,output3)
        output_m = wt1*output1+wt2*output2+wt3*output3
###添加ensemble label####
        ensembel_label = vote_ensemble(output1,output2,output3)
        ensembel_loss = self.contrast_loss(output,ensembel_label)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        supervised_loss = self.loss(output[:self.args.labeled_bs],label[:self.args.labeled_bs])
        output_ema_soft = softmax(output_ema,dim=1)
        output_soft = softmax(output,dim=1)
        if self.global_step<50:
            consistency_loss = torch.tensor(0.0)
            consistency_loss = consistency_loss.to(self.device)
            entropy_loss = torch.tensor(0.0)
            entropy_loss = entropy_loss.to(self.device)
        else:
            consistency_loss = torch.mean(
                (output_soft[self.args.labeled_bs:]-output_ema_soft)**2)
            entropy_loss = torch.mean(Entropy(output_soft))
            msoftmax = output_soft.mean(dim=0)
            gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss -= gentropy_loss
        consistency_weight = self.get_current_consistency_weight(self.global_step)
        unsupervised_loss = consistency_weight * consistency_loss
        all_loss = supervised_loss + unsupervised_loss + distill_loss/10000 + 0*entropy_loss/(img.shape[0]*100) + ensembel_loss
        self.update_ema_variables(self.model, self.ema_model, self.args.ema_decay, self.global_step)
        return {'loss':all_loss,'supervised_loss':supervised_loss,'unsupervised_loss':unsupervised_loss,'distill_loss':distill_loss,'entropy_loss':entropy_loss}
    def training_epoch_end(self, outputs):
        all_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()
        supervised_loss_epoch = torch.stack([x['supervised_loss'] for x in outputs]).mean()
        unsupervised_loss_epoch = torch.stack([x['unsupervised_loss'] for x in outputs]).mean()
        distill_loss_epoch = torch.stack([x['distill_loss'] for x in outputs]).mean()
        entropy_loss_epoch = torch.stack([x['entropy_loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',all_loss_epoch,prog_bar=True)
        self.log_dict({'train_supervised_loss_epoch':supervised_loss_epoch,\
                        'train_unsupervised_loss_epoch':unsupervised_loss_epoch,\
                        'train_distill_loss_epoch':distill_loss_epoch,\
                        'train_entropy_loss_epoch':entropy_loss_epoch})
        
    def get_current_consistency_weight(self,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)
    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
#TODO: ###############################多源域权重的融合(RIPU)######################
    def cal_src_wt(self,logits1,logits2,logits3):
        score1,score2,score3 = 0,0,0
        for i in range(len(logits1)):
            o1,o2,o3 = logits1[i:i + 1, :, :, :],logits2[i:i + 1, :, :, :],logits3[i:i + 1, :, :, :]
            score1,score2,score3 = RegionSelection_BBDA(o1)+score1, RegionSelection_BBDA(o2)+score2,RegionSelection_BBDA(o3)+score3
        wt1,wt2,wt3 = score1/(score1+score2+score3),score2/(score1+score2+score3),score3/(score1+score2+score3)
        return wt1,wt2,wt3
    
class Semi_Distill_DA_Unsuper(Basemodel):
    def __init__(self,args,model,ema_model,model1,model2,model3):
        super().__init__(args,model)
        self.ema_model = ema_model
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.contrast_loss = CrossEntropyLoss(ignore_index=-1)
    def training_step(self,batch,batch_idx):
        img,label = batch['img'],batch['label']
        ema_input = img[self.args.labeled_bs:]#无标签数据一致性计算
        noise = torch.clamp(torch.randn_like(
            ema_input) * 0.1, -0.2, 0.2)
        ema_input = ema_input + noise
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval(),self.ema_model.eval()
        with torch.no_grad():
            output_ema = self.ema_model(ema_input)
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)
            wt1,wt2,wt3 = self.cal_src_wt(output1,output2,output3)
        output_m = wt1*output1+wt2*output2+wt3*output3
###添加ensemble label####
        ensembel_label = vote_ensemble(output1,output2,output3)
        ensembel_loss = self.contrast_loss(output,ensembel_label)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        supervised_loss = self.loss(output[:self.args.labeled_bs],label[:self.args.labeled_bs])
        output_ema_soft = softmax(output_ema,dim=1)
        output_soft = softmax(output,dim=1)
        if self.global_step<50:
            consistency_loss = torch.tensor(0.0)
            consistency_loss = consistency_loss.to(self.device)
            entropy_loss = torch.tensor(0.0)
            entropy_loss = entropy_loss.to(self.device)
        else:
            consistency_loss = torch.mean(
                (output_soft[self.args.labeled_bs:]-output_ema_soft)**2)
            entropy_loss = torch.mean(Entropy(output_soft))
            msoftmax = output_soft.mean(dim=0)
            gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss -= gentropy_loss
        consistency_weight = self.get_current_consistency_weight(self.global_step)
        unsupervised_loss = consistency_weight * consistency_loss
        all_loss = 0*supervised_loss + unsupervised_loss + distill_loss/100000 + entropy_loss/(img.shape[0]*100) + ensembel_loss
        self.update_ema_variables(self.model, self.ema_model, self.args.ema_decay, self.global_step)
        return {'loss':all_loss,'supervised_loss':supervised_loss,'unsupervised_loss':unsupervised_loss,'distill_loss':distill_loss,'entropy_loss':entropy_loss}
    def training_epoch_end(self, outputs):
        all_loss_epoch = torch.stack([x['loss'] for x in outputs]).mean()
        supervised_loss_epoch = torch.stack([x['supervised_loss'] for x in outputs]).mean()
        unsupervised_loss_epoch = torch.stack([x['unsupervised_loss'] for x in outputs]).mean()
        distill_loss_epoch = torch.stack([x['distill_loss'] for x in outputs]).mean()
        entropy_loss_epoch = torch.stack([x['entropy_loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',all_loss_epoch,prog_bar=True)
        self.log_dict({'train_supervised_loss_epoch':supervised_loss_epoch,\
                        'train_unsupervised_loss_epoch':unsupervised_loss_epoch,\
                        'train_distill_loss_epoch':distill_loss_epoch,\
                        'train_entropy_loss_epoch':entropy_loss_epoch})
        
    def get_current_consistency_weight(self,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * ramps.sigmoid_rampup(epoch, self.args.consistency_rampup)
    def update_ema_variables(self,model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
#TODO: ###############################多源域权重的融合(RIPU)######################
    def cal_src_wt(self,logits1,logits2,logits3):
        score1,score2,score3 = 0,0,0
        for i in range(len(logits1)):
            o1,o2,o3 = logits1[i:i + 1, :, :, :],logits2[i:i + 1, :, :, :],logits3[i:i + 1, :, :, :]
            score1,score2,score3 = RegionSelection_BBDA(o1)+score1, RegionSelection_BBDA(o2)+score2,RegionSelection_BBDA(o3)+score3
        wt1,wt2,wt3 = score1/(score1+score2+score3),score2/(score1+score2+score3),score3/(score1+score2+score3)
        return wt1,wt2,wt3  
def vote_ensemble(logits1,logits2,logits3,threshold=0.9):
    logits1_softmax,logits2_softmax,logits3_softmax = softmax(logits1,dim=1),softmax(logits2,dim=1),softmax(logits3,dim=1)
    mask1,mask2,mask3 = logits1_softmax>threshold,logits2_softmax>threshold,logits3_softmax>threshold
    vote_mask = mask1*mask2 + mask1*mask3 + mask2*mask3
    logits1_softmax[~vote_mask],logits2_softmax[~vote_mask],logits3_softmax[~vote_mask] = -1,-1,-1
    ensemble_logits = logits1_softmax+logits2_softmax+logits3_softmax
    ensemble_label = torch.argmax(ensemble_logits,dim=1)
    ensemble_label = ensemble_label.unsqueeze(1)
    mask = torch.zeros_like(ensemble_label)
    for i,j in enumerate(vote_mask):
        mask[i] = j[:1]*j[1:]
    ensemble_label[~mask] = -1
    return ensemble_label.squeeze(1)
    