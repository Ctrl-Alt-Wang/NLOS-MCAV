# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from PIL import Image
import logging
import glob
from torch.utils.data import Dataset
import torch
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import os
from torch import optim
#from model.unet import UNet
from model import introvae
from model.danaet_maxpooling import DANET
from model.r2unet_swin import R2U_Net
from model.TransUnet import UNet_Attention_Transformer_Multiscale
from util.metrics import ssim
from util.metrics import PSNR
import time
from model.Phong import Phong
os.environ['CUDA_VISIBLE_DEVICE']='0'
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

parser = argparse.ArgumentParser()
# parser.add_argument('--datarootData', default=r"G:\CMY\DataSet\7-18output\ir")
# parser.add_argument('--datarootTarget', default=r"G:\CMY\DataSet\7-18output\vis")
# parser.add_argument('--datarootValData', default=r"G:\CMY\DataSet\7-18output\ir")
# parser.add_argument('--datarootValTarget', default=r"G:\CMY\DataSet\7-18output\vis")
# parser.add_argument('--datarootTestData', default=r"G:\CMY\DataSet\7-18output\ir")
# parser.add_argument('--datarootTestTarget', default=r"G:\CMY\DataSet\7-18output\vis")
##########################NEWDATA CARTON########################
# parser.add_argument('--datarootData', default=r"H:\cmy\train\25_band_carton_clean\visible")
# parser.add_argument('--datarootTarget', default=r"H:\cmy\train\25_band_carton_clean\sharp")
# parser.add_argument('--datarootValData', default=r"H:\cmy\valid\25_band_carton\visible")
# parser.add_argument('--datarootValTarget', default=r"H:\cmy\valid\25_band_carton\sharp")
# parser.add_argument('--datarootTestData', default=r"H:\cmy\valid\25_band_carton\visible")
# parser.add_argument('--datarootTestTarget', default=r"H:\cmy\valid\25_band_carton\sharp")
##########################NEWDATA SUPERMODEL########################
# parser.add_argument('--datarootData', default=r"H:\cmy\train\25_band_chaomo_clean\visible")
# parser.add_argument('--datarootTarget', default=r"H:\cmy\train\25_band_chaomo_clean\sharp")
# parser.add_argument('--datarootValData', default=r"H:\cmy\valid\25_band_chaomo\visible")
# parser.add_argument('--datarootValTarget', default=r"H:\cmy\valid\25_band_chaomo\sharp")
# parser.add_argument('--datarootTestData', default=r"H:\cmy\valid\25_band_chaomo\visible")
# parser.add_argument('--datarootTestTarget', default=r"H:\cmy\valid\25_band_chaomo\sharp")
#####################################new data stl###########################
parser.add_argument('--datarootData', default=r"D:\桌面\Nest-single\images\train\lr")
parser.add_argument('--datarootTarget', default=r"D:\桌面\Nest-single\images\train\label_lr")
parser.add_argument('--datarootValData', default=r"D:\桌面\Nest-single\images\valid\lr")
parser.add_argument('--datarootValTarget', default=r"D:\桌面\Nest-single\images\valid\lr_label")
parser.add_argument('--datarootTestData', default=r"D:\桌面\Nest-single\images\train\lr")
parser.add_argument('--datarootTestTarget', default=r"D:\桌面\Nest-single\images\train\label_lr")
# parser.add_argument('--datarootTestData', default=r"D:\桌面\100train\lr")
# parser.add_argument('--datarootTestTarget', default=r"D:\桌面\100train\lr_label")
#####################################public data supermodel###########################
# parser.add_argument('--datarootData', default=r"G:\YXD\supermodel\train\projection")
# parser.add_argument('--datarootTarget', default=r"G:\YXD\supermodel\train\target")
# parser.add_argument('--datarootValData', default=r"G:\YXD\supermodel\valid\projection")
# parser.add_argument('--datarootValTarget', default=r"G:\YXD\supermodel\valid\target")
# parser.add_argument('--datarootTestData', default=r"G:\YXD\supermodel\valid\projection")
# parser.add_argument('--datarootTestTarget', default=r"G:\YXD\supermodel\valid\target")
#####################################public data carton###########################
# parser.add_argument('--datarootData', default=r"G:\YXD\carton\train\projection")
# parser.add_argument('--datarootTarget', default=r"G:\YXD\carton\train\target")
# parser.add_argument('--datarootValData', default=r"G:\YXD\carton\valid\projection")
# parser.add_argument('--datarootValTarget', default=r"G:\YXD\carton\valid\target")
# parser.add_argument('--datarootTestData', default=r"G:\YXD\carton\valid\projection")
# parser.add_argument('--datarootTestTarget', default=r"G:\YXD\carton\valid\target")
parser.add_argument('--epoches',type=int,default=10000)
parser.add_argument('--batchsize',type=int,default=1)
parser.add_argument('--num_workers',type=int,default=0)
parser.add_argument('--name',type=str,default='publicdata_phong_wb')
parser.add_argument('--model_name',type=str,default='WB')
parser.add_argument('--learning_rate',type=float,default=1e-5)
parser.add_argument('--sizeImage',type=int,default=256)
parser.add_argument('--trainContinue',type=bool,default=False)
parser.add_argument('--val_freq',type=int,default=5000)
parser.add_argument('--display_freq',type=int,default=200)
parser.add_argument('--test_checkpoints_load_dir', type=str, default='./checkpoints/publicdata_phong_WB')
parser.add_argument('--which_epoch', type=str, default='', help='which epoch to load? set to latest to use latest cached model')
# Press the green button in the gutter to run the script.
class Basicdataset(Dataset):
    def __init__(self,img_dir,label_dir,sizeImage,scale=1.0,mask_suffix=True):
        self.img_dir=img_dir
        self.label_dir=label_dir
        self.scale=scale
        self.mask_suffix=mask_suffix
        self.sizeImage=sizeImage
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
        #a = listdir(img_dir)

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

def loadNetwork(model,modelname,networkLabel,path,epoch=None):
    # filename=modelname+'_'+networkLabel+'.pth'
    filename=modelname+'_'+networkLabel+'.pth'
    pathSave=os.path.join(path,filename)
    state_dict = torch.load(pathSave, map_location=torch.device('cuda:0'))
    model.load_state_dict(state_dict['model'])
    print('loade succesfully')
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def tensor2im(image_tensor, imtype=np.uint8):
	image_numpy = image_tensor[0].cpu().float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	return image_numpy.astype(imtype)

if __name__ == '__main__':
    args = parser.parse_args()
    all_imgs_path_train = glob.glob(args.datarootData + '\*.bmp')
    all_target_path_train = glob.glob(args.datarootTarget + '\*.bmp')
    all_imgs_path_test = glob.glob(args.datarootTestData + '\*.bmp')
    all_target_path_test = glob.glob(args.datarootTestTarget + '\*.bmp')
    timeRun = time.time()
    dataset_test = Basicdataset(all_imgs_path_test,all_target_path_test,args.sizeImage)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batchsize, shuffle=False,#原本是dataset
        num_workers=args.num_workers, drop_last=True)
    model=Phong()


    loadNetwork(model=model,modelname=args.model_name,networkLabel=args.which_epoch,path=args.test_checkpoints_load_dir,epoch=None)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.MSELoss()
    step=0
    idx=0
    psnr_sum=0
    ssim_sum=0
    txtName = "test_loss.txt"
    picturedir_test = str("./test_result/" + args.name)
    if not os.path.exists(picturedir_test):
        os.makedirs(picturedir_test)
    texdir = picturedir_test + '/' + txtName
    f = open(texdir, "w")
# See PyCharm help at https://www.jetbrains.com/help/pychF
    for img,label,name in iter(dataloader_test):
        img = img.to(device)
        label = label.to(device)
        model.eval()
        pred = model(img)
        pred_img = tensor2im(pred.data)
        label_img = tensor2im(label.data)
        ssimMetric = ssim(pred_img, label_img)
        psnrMetric = PSNR(pred_img, label_img)
        ssim_sum += ssimMetric
        psnr_sum += psnrMetric
        idx += 1
        picturedir_path = picturedir_test + '/' + str(name[0]) + '.bmp'
        transtensortoimg(pred, picturedir_path)
        context = str(name[0])+'  SSIM on Train = %f;PSNR on Train = %f' %(ssimMetric, psnrMetric)+'\n'
        print(context)
        f.write(context)
    Average_context = 'Average_PSNR = %.3f ' % (psnr_sum / idx) + 'Average_SSIM=%.3f' % (
                    ssim_sum / idx) + '\n'

    f.write(Average_context)
    print(Average_context)


