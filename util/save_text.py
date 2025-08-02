import csv
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def save_data(steps, losses, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #writer.writerow(['Steps', 'Losses'])  # 写入标题行
        #for step, loss in zip(steps, losses):
        writer.writerow([steps, losses])  # 写入步数和损失数据

class plotdataer():
    def save(self,steps, losses, filename):
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['Steps', 'Losses'])  # 写入标题行
            # for step, loss in zip(steps, losses):
            writer.writerow([steps, losses])  # 写入步数和损失数据
    def plot(self,*args):
        num_parameters = len(args)
        if num_parameters==1:
            df1 = pd.read_csv(args[0],names=['Steps', 'Losses'])
            steps = df1['Steps']
            losses = df1['Losses']
            fig, ax = plt.subplots()
            ax.plot(steps, losses, linestyle='-', color='r', label='UNET')
            ax.legend()
            ax.set_title('Step vs. Loss')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Losses')

            plt.savefig('publicdata_supermodel_transformer.png')
            plt.show()
        if num_parameters == 2:
            df1 = pd.read_csv(args[0],names=['Steps', 'Losses'])
            df2 = pd.read_csv(args[1],names=['Steps', 'Losses'])
            # 提取步数和损失列
            steps = df1['Steps']
            losses = df1['Losses']
            steps2 = df2['Steps']
            losses2 = df2['Losses']
            fig, ax = plt.subplots()
            # 绘制折线图
            ax.plot(steps, losses, linestyle='-', color='r', label='UNET')
            # 绘制折线
            ax.plot(steps2, losses2, linestyle='--', color='b', label='R2UNET')
            # 设置图例
            ax.legend()
            # 设置标题和坐标轴标签
            ax.set_title('Step vs. Loss')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Losses')
            # 显示图形
            plt.savefig('publicdata_carton_r2unet.png')
            plt.show()


def plot_data(filename1,filename2,filename3):
    # 读取CSV文件
    df1 = pd.read_csv(filename1,names=['Steps', 'Losses'])
    df2 = pd.read_csv(filename2,names=['Steps', 'Losses'])
    df3 = pd.read_csv(filename3,names=['Steps', 'Losses'])

    # 提取步数和损失列
    steps = df1['Steps']
    losses = df1['Losses']
    steps2 = df2['Steps']
    losses2 = df2['Losses']
    steps3 = df3['Steps']
    losses3 = df3['Losses']
    fig, ax = plt.subplots()
    # 绘制折线图
    ax.plot(steps, losses,linestyle='-',color='r',label='UNET')
    # 绘制折线
    ax.plot(steps2, losses2, linestyle='--', color='b', label='R2UNET')
    ax.plot(steps3, losses3, linestyle='--', color='g', label='SEGNET')
    # 设置图例
    ax.legend()

    # 设置标题和坐标轴标签
    ax.set_title('Step vs. Loss')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Losses')



    #显示图形

    plt.savefig('IR_PEOPLE_r2unet.png')
    plt.show()


if __name__ == '__main__':
    name1 = r"G:\CMY\PycharmProject\self_net_family\checkpoints\publicdata_carton_unet\publicdata_carton_unet500"
    name2 = r"G:\CMY\PycharmProject\self_net_family\checkpoints\publicdata_carton_r2unet\publicdata_carton_r2unet500"
    name3 =r"G:\CMY\PycharmProject\self_net_family\checkpoints\IR_redboard_people_r2unet\IR_redboard_people_r2unet"
    #plot_data(name1,name2,name3)
    ploter=plotdataer()
    #ploter.plot(name1,name2)
    ploter.plot(name3)