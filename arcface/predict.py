import matplotlib.pyplot as plt
from PIL import Image
import os
from arcface import Arcface
import torch
import numpy as np
model = Arcface()
if __name__ == "__main__":
    path_name='/data/arcface-pytorch-main/100_1'
    def read_path(path_name):
        best_score = 0.5
        label = 'none'
        image_2 = input('input image filename:')
        image_2 = Image.open(image_2)
        for dir in os.listdir(path_name):
            # 从初始路径开始叠加，合并成可识别的操作路径
            for img in os.listdir(os.path.join(path_name,dir)):
                full_path = os.path.abspath(os.path.join(path_name, dir,img))
                if img.endswith('.jpg'):
                    image_1 = Image.open(full_path)
                    distances = model.detect_image(image_1, image_2)
                    if distances < best_score:
                        best_score = distances
                        label=dir
                    else:
                        pass
        print(1 - best_score, label)
        return label,best_score
        plt.subplot(1, 1, 1)
        # plt.imshow(np.array(image_1))
        # plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'probability:%.3f' % (1-best_score), ha='center', va='bottom', fontsize=11)
        # plt.show()
    read_path(path_name)
