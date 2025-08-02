# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import glob
import os.path
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
parser=argparse.ArgumentParser()
parser.add_argument('--datarootData',default=r'H:\cmy\datasets\Carton\valid\blur')
parser.add_argument('--saveData',default=r'H:\cmy\datasets\Carton\valid\blur')
parser.add_argument('--name',default='clutter_background')
parser.add_argument('--number_single_value',type=int,default=50)
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

class Basicdataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir=img_dir
    def __getitem__(self, item):
        name1=os.path.splitext(self.img_dir[item])
        name = os.path.splitext(self.img_dir[item])[0]
        name=os.path.split(name)[1]
        img=self.img_dir[item]
        assert os.path.isfile(img), '%s is not a valid directory' % dir
        img = Image.open(img)
        img = np.array(img)
        return {'data': img,  'name': name}
    def __len__(self):
        return len(self.img_dir)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parser.parse_args()

    all_imgs_path_svd = glob.glob(args.datarootData + '\*.png')
    dataset_svd = Basicdataset(all_imgs_path_svd)
    dataset_loder_svd = torch.utils.data.DataLoader(
        dataset_svd, batch_size=1, shuffle=False,  # 原本是dataset
        num_workers=1, drop_last=True)
    print('dataloader_train')
    step = 0
    number_single_value=args.number_single_value
    picturedir_svd = args.datarootData + '\\' + str(args.number_single_value)
    if not os.path.exists(picturedir_svd):
        os.makedirs(picturedir_svd)
    for i, data in enumerate(dataset_loder_svd):

        step = step + 1
        # 将PyTorch的张量转换为NumPy数组
        img = data['data'].numpy()
        B= img[0, :, :, 0]
        G= img[0, :, :, 1]
        R = img[0, :, :, 2]
        # 进行奇异值分解
        U_B, S_B, V_B = np.linalg.svd(B)
        # U、S和V分别是左奇异矩阵、奇异值和右奇异矩阵
        # print("Left Singular Vectors (U):\n", U_B)
        # print("Singular Values (S):\n", S_B)
        # print("Right Singular Vectors (V):\n", V_B)
        # 保留前n个奇异值，其他置零
        S_prime_truncated = np.diag(S_B[:number_single_value])
        # 重构原始图像矩阵
        reconstructed_matrix_B = np.dot(U_B[:, :number_single_value], np.dot(S_prime_truncated, V_B[:number_single_value, :]))

        U_G, S_G, V_G = np.linalg.svd(G)
        # U、S和V分别是左奇异矩阵、奇异值和右奇异矩阵
        # print("Left Singular Vectors (U):\n", U_G)
        # print("Singular Values (S):\n", S_G)
        # print("Right Singular Vectors (V):\n", V_G)
        # 保留前n个奇异值，其他置零
        S_prime_truncated = np.diag(S_G[:number_single_value])
        # 重构原始图像矩阵
        reconstructed_matrix_G = np.dot(U_G[:, :number_single_value], np.dot(S_prime_truncated, V_G[:number_single_value, :]))

        U_R, S_R, V_R = np.linalg.svd(R)
        # U、S和V分别是左奇异矩阵、奇异值和右奇异矩阵
        # print("Left Singular Vectors (U):\n", U_R)
        # print("Singular Values (S):\n", S_R)
        # print("Right Singular Vectors (V):\n", V_R)
        # 保留前n个奇异值，其他置零
        S_prime_truncated = np.diag(S_R[:number_single_value])
        # 重构原始图像矩阵
        reconstructed_matrix_R = np.dot(U_R[:, :number_single_value], np.dot(S_prime_truncated, V_R[:number_single_value, :]))
        # 将矩阵转换为图像
        result = np.stack((reconstructed_matrix_B, reconstructed_matrix_G,reconstructed_matrix_R),axis=0)
        result = result.astype(np.uint8)
        result = np.transpose(result)
        reconstructed_image = Image.fromarray(result)
        reconstructed_image=reconstructed_image.rotate(270).transpose(Image.FLIP_LEFT_RIGHT)

        # 显示或保存重构后的图像
        # reconstructed_image.show()
        # 重构后的图像保存到文件

        picturedir = os.path.join(picturedir_svd,str(data['name'][0]) + '.png')
        reconstructed_image.save(picturedir)
        print(str(data['name'][0])+'saved')



