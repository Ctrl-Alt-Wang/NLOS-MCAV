 # This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys 

import numpy as np
import pytorch_msssim
import torchvision.transforms.functional
from PIL import Image
import torchvision.transforms as transforms
import logging
import glob
from torch.utils.data import Dataset
import torch
import argparse
from util.save_text import save_data
import torch.nn as nn
import os
from torch import optim
from model.r2unet import R2U_Net#改完模型记得改模型存储路径
# from model.attention_unet import Attention_block
# from model.TRO_Net_OT import TRO_Net
# from model.restormer_arch import Restormer
from util.metrics import ssim
from util.metrics import PSNR


import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

parser = argparse.ArgumentParser()
#####################################public data carton###########################


parser.add_argument('--datarootData', default="Data/MG1500/train/BR") 
parser.add_argument('--datarootTarget', default="Data/MG1500/train/GT")
parser.add_argument('--datarootValData', default="Data/MG1500/test/BR")
parser.add_argument('--datarootValTarget', default="Data/MG1500/test/GT")
parser.add_argument('--datarootTestData', default="Data/MG1500/test/BR")
parser.add_argument('--datarootTestTarget', default="Data/MG1500/test/GT")


# parser.add_argument('--datarootData', default="Data/train/vis/BR") 
# parser.add_argument('--datarootTarget', default="Data/train/vis/GT")
# parser.add_argument('--datarootValData', default="Data/test/vis_test/BR")
# parser.add_argument('--datarootValTarget', default="Data/test/vis_test/GT")
# parser.add_argument('--datarootTestData', default="Data/test/vis_test/BR")
# parser.add_argument('--datarootTestTarget', default="Data/test/vis_test/GT")

parser.add_argument('--epoches',type=int,default=200)
parser.add_argument('--batchsize',type=int,default=1)
parser.add_argument('--num_workers',type=int,default=0)
parser.add_argument('--name',type=str,default='NLOS-ST_MG')#最后一个"_"后要写模型名
parser.add_argument('--learning_rate',type=float,default=1e-5)
parser.add_argument('--sizeImage',type=int,default=256)#其他都是256
parser.add_argument('--trainContinue',type=bool,default=False)
parser.add_argument('--val_freq',type=int,default=15000)
parser.add_argument('--display_freq',type=int,default=200)
parser.add_argument('--train_checkpoints_load_dir', type=str, default='')
parser.add_argument('--which_epoch', type=str, default='', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--epoch_start', type=int, default='0')
# Press the green button in the gutter to run the script.
class Basic_train_dataset(Dataset):
    def __init__(self,img_dir,label_dir,sizeImage,scale=1.0,mask_suffix=True):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.scale=scale
        self.mask_suffix=mask_suffix
        self.sizeImage=sizeImage
        #a = listdir(img_dir)
        #################验证集数据增强##########################
        # transform_list = [transforms.RandomHorizontalFlip(),
        #                   transforms.RandomRotation(25),
        #                   transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5),
        #                                        (0.5, 0.5, 0.5))]
        transform_list = [
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        # for file in listdir(self.img_dir):
        #     self.ids=splitext(file)[0]
        logging.info('image_number{len(self.ids)}')

    def __getitem__(self, index):
        nama1=self.img_dir[index]
        name=os.path.splitext(self.img_dir[index])[0]
        name=os.path.split(name)[1]
        img = self.img_dir[index]
        assert os.path.isfile(img), '%s is not a valid directory' % dir
        label = self.label_dir[index]
        assert os.path.isfile(label), '%s is not a valid directory' % dir
        img = Image.open(img)
        label = Image.open(label)
        # img=img.convert('L')
        # label = label.convert('L')
        img = img.resize((self.sizeImage,self.sizeImage),Image.BICUBIC)
        label = label.resize((self.sizeImage, self.sizeImage), Image.BICUBIC)
        img = self.transform(img)
        label = self.transform(label)
        return img, label,name
    def __len__(self):
        return len(self.img_dir)
class Basic_valid_dataset(Dataset):
    def __init__(self,img_dir,label_dir,sizeImage,scale=1.0,mask_suffix=True):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.scale=scale
        self.mask_suffix=mask_suffix
        self.sizeImage=sizeImage
        #a = listdir(img_dir)
        transform_list = [
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        # for file in listdir(self.img_dir):
        #     self.ids=splitext(file)[0]
        logging.info('image_number{len(self.ids)}')

    def __getitem__(self, index):
        nama1=self.img_dir[index]
        name=os.path.splitext(self.img_dir[index])[0]
        name=os.path.split(name)[1]
        img = self.img_dir[index]
        assert os.path.isfile(img), '%s is not a valid directory' % dir
        label = self.label_dir[index]
        assert os.path.isfile(label), '%s is not a valid directory' % dir
        img = Image.open(img)
        label = Image.open(label)
        img = img.resize((self.sizeImage,self.sizeImage),Image.BICUBIC)
        label = label.resize((self.sizeImage, self.sizeImage), Image.BICUBIC)
        img = self.transform(img)
        label = self.transform(label)
        return img, label,name
    def __len__(self):
        return len(self.img_dir)
def transtensortoimg(img,pathtosave,imtype=np.uint8):
    predpic=img[0].detach().cpu().numpy()
    predpic=(np.transpose(predpic,(1,2,0))+1)/2.0*255
    predpic=predpic.astype(imtype)
    predpic=Image.fromarray(predpic)
    predpic.save(pathtosave)
def optimizer_mix(model,pred,label):
    encoder=model.encoder
    decoder=model.decoder
    optimizer_encoder=optim.Adam(encoder.parameters(),args.learning_rate)
    optimizer_decoder=optim.Adam(decoder.parameters(),args.learning_rate)
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()
    loss=criterion(pred,label)
    loss.backward()
    optimizer_encoder.step()
    optimizer_decoder.step()
    return loss
def saveNetwork(model,networkLabel,path,epoch=None):
    filename=str(networkLabel)+'_'+str(epoch)+'.pth'
    pathSave=os.path.join(path,filename)
    # torch.save(model.state_dict(),pathSave)
    opt_state = {'epoch': epoch, 'model': None}
    opt_state['model'] = model.state_dict()
    torch.save(opt_state, pathSave)
def loadNetwork(model,networkLabel,path,epoch=None):
    filename=str(networkLabel)+'_'+str(epoch)+'.pth'
    pathSave=os.path.join(path,filename)
    state_dict = torch.load(pathSave)
    epoch=state_dict['epoch']
    model.load_state_dict(state_dict['model'])
    print('%s,loade succesfully'%(filename))


def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor[0].cpu().float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	return image_numpy.astype(imtype)

if __name__ == '__main__':
    print_hi("Make a little progress every day")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICE'] = args.gpu_id
    all_imgs_path_train = glob.glob(args.datarootData + '/*.png')
    all_target_path_train = glob.glob(args.datarootTarget + '/*.png')
    all_imgs_path_valid = glob.glob(args.datarootValData + '/*.png')
    all_target_path_valid = glob.glob(args.datarootValTarget + '/*.png')
    print("Number of images in training data:", len(all_imgs_path_train))
    print("Number of target images in training data:", len(all_target_path_train))
    print("Number of images in validation data:", len(all_imgs_path_valid))
    print("Number of target images in validation data:", len(all_target_path_valid))



    ######################实验室双目相机采集数据##########################
    # all_imgs_path_train = glob.glob(args.datarootData + '\*.bmp')
    # all_target_path_train = glob.glob(args.datarootTarget + '\*.bmp')
    # all_imgs_path_valid = glob.glob(args.datarootValData + '\*.bmp')
    # all_target_path_valid = glob.glob(args.datarootValTarget + '\*.bmp')
    timeRun = time.time()
    dataset_train = Basic_train_dataset(all_imgs_path_train,all_target_path_train,args.sizeImage)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batchsize, shuffle=False,#原本是dataset
        num_workers=args.num_workers, drop_last=True)

    dataset_valid=Basic_valid_dataset(all_imgs_path_valid,all_target_path_valid,args.sizeImage)
    dataloader_valid=torch.utils.data.DataLoader(
        dataset_valid,batch_size=args.batchsize,shuffle=False,
        num_workers=args.num_workers,drop_last=True)
    model = R2U_Net()
    #model = _DAHead()
    #model = IntroAE()
    #model = UNet_Attention_Transformer_Multiscale()
    #model = SwinTransformer()
    model_name = args.name
    model_name = model_name.split("_")[-1]
    #model = TRO_Net()
    #model = DaNet._DAHead(3,3)
    epoch_start=1
    dir_checkpoint = str('./checkpoints/' + args.name + '/')
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    if args.trainContinue:
        loadNetwork(model=model,networkLabel=model_name,path=args.train_checkpoints_load_dir,epoch=args.which_epoch)
    device = torch.device("cuda:%s" % (args.gpu_id) if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    # for name,parameter in model.parameter():
    #     if parameter.requires_grad:
    #print(list(model.parameters()))
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.MSELoss()
    Entropy_loss = nn.CrossEntropyLoss()
    SSIM_loss = pytorch_msssim.SSIM
    step=0
    train_step=[]
    train_loss=[]
# See PyCharm help at https://www.jetbrains.com/help/pychF
    for epoch in range(args.epoch_start,args.epoches):
        for img,label,name in iter(dataloader_train):
            step = step + 1
            img = img.to(device)
            label = label.to(device)
            pred = model(img)
            #loss = optimizer_mix(model,pred,label)
            # loss = criterion(pred, label)
            # loss_1 = Entropy_loss(pred,label)
            pred_img = tensor2im(pred.data)
            label_img = tensor2im(label.data)
            # loss_2 = ssim(pred_img,label_img)
            # loss_2 = torch.tensor(loss_2)
            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            # grad_scaler.scale(loss).backward()
            # grad_scaler.step(optimizer)
            # grad_scaler.update()

            picturedir_valid = str("./valid_result/"+args.name)
            #filedir = os.path.join('./checkpoints/'+ args.name+'/', txtName)
            filedir = str('./checkpoints/' + args.name)
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            save_loss = float(loss)
            train_step.append(step)
            train_loss.append(round(save_loss,3))
            save_data(step,round(save_loss,3),filedir+'/'+args.name)
            if not os.path.exists(picturedir_valid):
                os.makedirs(picturedir_valid)
            if step % args.display_freq == 0:
                pred_img = tensor2im(pred.data)
                label_img = tensor2im(label.data)
                ssimMetric = ssim(pred_img, label_img)
                psnrMetric = PSNR(pred_img, label_img)
                print('step=%d;SSIM on Train = %f;PSNR on Train = %f' %
                      (step,ssimMetric,psnrMetric))
            #print('step'+str(step)+',loss %f'%(loss))
            if step % args.val_freq == 0:
                ssim_sum=0
                psnr_sum=0
                idx=0
                txtName = "val_loss.txt"
                texdir = filedir + '/' + txtName
                f = open(texdir, "a+")
                for img,label,name in iter(dataloader_valid):
                    img=img.to(device)
                    label=label.to(device)
                    model.eval()
                    pred=model(img)
                    pred_img=tensor2im(pred.data)
                    label_img=tensor2im(label.data)
                    ssim_sum += ssim(pred_img,label_img)
                    psnr_sum += PSNR(pred_img,label_img)
                    idx += 1
                    picturedir_path=picturedir_valid+'/'+str(name[0])+'.bmp'
                    transtensortoimg(pred,picturedir_path)
                    saveNetwork(model=model,networkLabel=model_name,path=dir_checkpoint,epoch=epoch)
                    saveNetwork(model=model, networkLabel=model_name, path=dir_checkpoint,epoch='latest')
                Average_context = 'Epoch=%d' %(epoch) + 'Average_PSNR = %.3f ' % (psnr_sum / idx) + 'Average_SSIM=%.3f' % (ssim_sum / idx) + '\n'
                print(Average_context)
                f.write(Average_context)

