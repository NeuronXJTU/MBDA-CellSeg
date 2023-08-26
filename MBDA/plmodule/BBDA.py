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
import argparse
import monai
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
import time
post_pred = Compose(
    [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
)
post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])
# model_path = '/media/oem/sda21/wxg/work_dirs/bbda/checkpoints'
# os.makedirs(model_path, exist_ok=True)
# run_id = datetime.now().strftime("%Y%m%d-%H%M")

# shutil.copyfile(
#     __file__, join(model_path, run_id + "_" + os.path.basename(__file__))
# )
def soft_difficult(logits1,logits2,logits3):
    a,b,c = torch.softmax(logits1, dim=1), torch.softmax(logits2, dim=1), torch.softmax(logits3, dim=1)
    a,b,c = a.unsqueeze(1),b.unsqueeze(1),c.unsqueeze(1)
    d = torch.cat((a,b,c), dim=1) #(batch,3,2,H,W)
    max_p, _ = torch.max(d, dim=2)#(B,M,H,W)
    mask = max_p>0.7
    sum_mask = torch.sum(mask, dim=1) # 我们认为对于有的模型的困难样本可以抛弃(B,H,W)
    mask = mask.unsqueeze(2).repeat(1,1,2,1,1) # (B,M,2,H,W)
    a_new = a*mask[:,0,:,:,:].unsqueeze(1)
    b_new = b*mask[:,1,:,:,:].unsqueeze(1)
    c_new = c*mask[:,2,:,:,:].unsqueeze(1)
    out_new = a_new+b_new+c_new
    old = a+b+c
    out_new[out_new==0] = old[out_new==0]
    sum_mask[sum_mask==0] = 3
    sum_mask = sum_mask.unsqueeze(1).repeat(1,2,1,1)
    out_new = out_new.squeeze(1)/sum_mask
    return out_new
    
class Basemodel(pl.LightningModule):
    def __init__(self,args,model=None,pretrained_model=None):
        super().__init__()
        self.model = model
        self.pretrained_model = pretrained_model
        self.args = args
        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice = DiceMetric(
            include_background=False, reduction="none", get_not_nans=True
            )
        self.hd95 = HausdorffDistanceMetric(percentile=95,reduction="mean",get_not_nans=False)

        self.assd = SurfaceDistanceMetric(reduction="mean",get_not_nans=False,symmetric=True)


    def forward(self, x):
        output = self.model(x)
        if self.pretrained_model is not None:
            with torch.no_grad():
                pretrain_output = self.pretrained_model(x)
            return output,pretrain_output
        else:
            return output
    def val_forward(self, x):
        with torch.no_grad():
            output = self.model(x)
        return output
    def slide_window_forward(self,inputs):
        outputs = sliding_window_inference(inputs, roi_size=self.args.roi_size, sw_batch_size=self.args.sw_batch_size, predictor=self.model)
        outputs = [post_pred(i) for i in decollate_batch(outputs)]
        return outputs
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), self.args.initial_lr)
        return optimizer
    def validation_step(self, batch, batch_idx):
        img,label = batch['img'],batch['label']
        val_labels_onehot = one_hot(label, self.args.num_class)
        val_outputs = self.val_forward(img)
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels_onehot = [
            post_gt(i) for i in decollate_batch(val_labels_onehot)
        ]
        self.dice(val_outputs,val_labels_onehot)
        self.hd95(val_outputs,val_labels_onehot)
        self.assd(val_outputs,val_labels_onehot)
        if batch_idx == 0:
            self.logger.experiment.add_images(
                "val_image", img[0], self.current_epoch,dataformats='CHW'
            )
            lab = label[0,0]
            lab[lab>0] = 255
            self.logger.experiment.add_images(
                "val_label", lab, self.current_epoch, dataformats="HW"
            )
            self.logger.experiment.add_images(
                "val_output", val_outputs[0][0], self.current_epoch, dataformats="HW"
            )
    def validation_epoch_end(self, outputs):
        dice = self.dice.aggregate(reduction='mean_batch')
        # hd95 = self.hd95.aggregate(reduction='mean_batch')
        # assd = self.assd.aggregate(reduction='mean_batch')
        hd95 = self.hd95.aggregate().item()
        assd = self.assd.aggregate().item()
        self.dice.reset(),self.hd95.reset(),self.assd.reset()
        dice = dice[0][0]
        # hd95 = hd95[0][0]
        # assd = assd[0][0]
        self.log('val_dice',dice,prog_bar=True)
        self.log_dict({'val_hd95':hd95,'val_assd':assd})
    def configure_callbacks(self):
        return [pl.callbacks.ModelCheckpoint(
            monitor='val_dice',
            filename='best_model_{epoch:04d}_{val_dice:.4f}',
            save_top_k=1,
            mode='max',
            save_last=True
        )]
class Pretrain(Basemodel):
    def __init__(self,args,model):
        super().__init__(args,model)
        self.pretrain_loss = DiceCELoss(to_onehot_y=True,softmax=True)
        self.diceloss = DiceLoss(to_onehot_y=True)
        self.start_time = time.time()
    def training_step(self, batch, batch_idx):
        img,label = batch['img'],batch['label']
        output = self(img)
        pretrain_loss = self.pretrain_loss(output,label)
        self.log('train_loss',pretrain_loss.item())
        return pretrain_loss
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',avg_loss,prog_bar=True)
    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            self.start_time = time.time()
        else:
            time_per_epoch = (time.time()-self.start_time)
            self.start_time = time.time()
            print(f'epoch {self.current_epoch} time:{time_per_epoch}')
        

class Distill(Basemodel):
    def __init__(self,args,model,pretrained_model):
        super().__init__(args,model,pretrained_model)
    def training_step(self, batch, batch_idx):
        img,_ = batch['img'],batch['label']
        output,pretrain_output = self.forward(img)
        distill_loss = softmax_kl_loss(output,pretrain_output)/self.args.batch_size
        self.log('train_loss',distill_loss.item())
        return distill_loss
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',avg_loss,prog_bar=True)
class ThreeDistill(Basemodel):
    def __init__(self,args,model,model1,model2,model3):
        super().__init__(args,model)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.ensemble = VoteEnsemble(num_classes=args.num_class)
    def training_step(self,batch,batch_idx):
        img,_ = batch['img'],batch['label']
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval()
        with torch.no_grad():
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)
        output1_soft, output2_soft, output3_soft = softmax(output1,dim=1), softmax(output2,dim=1), softmax(output3,dim=1)
        output_mean = (output1_soft + output2_soft + output3_soft) / 3
        output_m = (output1+output2+output3)/3
        # ###add pseudo label
        # p_label = torch.argmax(softmax(output_mean,dim=1),dim=1)
        # p_label = p_label.unsqueeze(1)
        # dice_celoss = self.loss(output,p_label)
        # ###end
        output_soft = softmax(output,dim=1)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        entropy_loss = torch.mean(Entropy(output_soft))
        msoftmax = output_soft.mean(dim=0)
        gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        # loss = distill_loss + entropy_loss/img.shape[0]+ dice_celoss
        # if self.global_step < 3:
        #     loss = distill_loss
        # else:
        #     loss = distill_loss + entropy_loss/(img.shape[0]*100) ### NO dice_celoss
        # loss = distill_loss
        loss = distill_loss + entropy_loss/(img.shape[0]*100)
        return loss
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',avg_loss,prog_bar=True)


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
        isave = name
        pth = os.path.join('KIRC_uda',isave)
        io.imsave(pth,mask)
class ThreeDistill_New_DisMethod(Basemodel):
    def __init__(self,args,model,model1,model2,model3):
        super().__init__(args,model)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.ensemble = VoteEnsemble(num_classes=args.num_class)
        self.consensus_knowledge = torch.zeros((args.num_class,args.num_class))
    def training_step(self,batch,batch_idx):
        img,_ = batch['img'],batch['label']
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval()
        with torch.no_grad():
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)

        output_m = (output1+output2+output3)/3
        output_soft = softmax(output,dim=1)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        entropy_loss = torch.mean(Entropy(output_soft))
        msoftmax = output_soft.mean(dim=0)
        gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss

        loss = distill_loss + entropy_loss/(img.shape[0]*100) ### NO dice_celoss
        return loss
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',avg_loss,prog_bar=True)

class Semi_Distill_DA(Basemodel):
    def __init__(self,args,model,ema_model,model1,model2,model3):
        super().__init__(args,model)
        self.ema_model = ema_model
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
    def training_step(self,batch,batch_idx):
        img,label = batch['img'],batch['label']
        labeled_label = label[:self.args.labeled_bs]
        labeled_img = img[:self.args.labeled_bs]
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
        all_loss = supervised_loss + unsupervised_loss + distill_loss/10000 + entropy_loss/(img.shape[0]*100)
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
        
class BETA(Basemodel):
    def __init__(self,args,model,model1,model2,model3):
        super().__init__(args,model)
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.all_psl = {}
        self.all_div = {}

    def training_step(self,batch,batch_idx):
        img,_,img_name= batch['img'],batch['label'],batch['idx']
        output = self.model(img)
        self.model1.eval(),self.model2.eval(),self.model3.eval()
        with torch.no_grad():
            output1 = self.model1(img)
            output2 = self.model2(img)
            output3 = self.model3(img)
        output_m = (output1+output2+output3)/3
        output_soft = softmax(output,dim=1)
        output_hard = torch.argmax(output_soft,dim=1)

        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        entropy_loss = torch.mean(Entropy(output_soft))
        msoftmax = output_soft.mean(dim=0)
        gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        loss = distill_loss + entropy_loss/(img.shape[0]*100)
        return loss
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss_epoch',avg_loss,prog_bar=True)

        
    def test_step(self,batch,batch_idx):
        img,_,img_name= batch['img'],batch['label'],batch['idx']
        output1 = self.model1(img)
        output2 = self.model2(img)
        output3 = self.model3(img)
        output_m = (output1+output2+output3)/3
        #create a dictionary of all the psl
        output_soft = softmax(output_m,dim=1)
        hard_output = torch.argmax(output_soft,dim=1)
        self.all_psl.update({img_name[i]:hard_output[i].cpu().numpy() for i in range(img.shape[0])})
    def on_test_end(self):
        #save the dictionary
        root = 'tmp_files'
        pkl_path = os.path.join(root,'psl.pkl')
        with open(pkl_path,'wb') as f:
            pickle.dump(self.all_psl,f)
            
        
        

class FrontierDistill(Basemodel):
    def __init__(self,args,model,model1):
        super().__init__(args,model)
        self.model1 = model1
    def training_step(self,batch,batch_idx):
        img,_ = batch['img'],batch['label']
        output = self.model(img)
        self.model1.eval()
        with torch.no_grad():
            output1 = self.model1(img)
        # output_m = (output+output1)/2
        if self.global_step<50:
            output_m = output1
        elif self.global_step<100:
            src_output = output.clone().detach()
            output_m = output1*0.5+src_output*0.5
        else:
            src_output = output.clone().detach()
            output_m = output1*0.3+src_output*0.7
        output_soft = softmax(output,dim=1)
        distill_loss = softmax_kl_loss(output,output_m)/img.shape[0]
        entropy_loss = torch.mean(Entropy(output_soft))
        msoftmax = output_soft.mean(dim=0)
        gentropy_loss = -torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        entropy_loss -= gentropy_loss
        loss = distill_loss + entropy_loss/(img.shape[0]*100)
        return loss