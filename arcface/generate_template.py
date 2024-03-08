import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from arcface import Arcface
import torch
import numpy as np
model = Arcface()
if __name__ == "__main__":
    path_name= 'data'
    map_embeddings = {}
    def read_path(path_name):
        csv_path = 'csv'
        path = os.listdir(path_name)
        path.sort()
        list0 = []
        list1 = []
        for dir in path:
            # 从初始路径开始叠加，合并成可识别的操作路径
            #print(dir)
            label = dir
            for img in os.listdir(os.path.join(path_name, dir)):

                full_path = os.path.abspath(os.path.join(path_name, dir,img))

                if img.endswith('.jpg'):
                    image = Image.open(full_path)
                    embedding = model.get_features(image)
                    list0.append(embedding)
                    list1.append(label)
        for embedding, label in zip(list0, list1):
            if label not in map_embeddings:
                map_embeddings[label] = [embedding]
            else:
                map_embeddings[label].append(embedding)

        if not os.path.exists(csv_path):
            os.mkdir(csv_path)
        for label, emb in map_embeddings.items():
            # print(label)
            # print(emb)
            emb = np.array(emb)
            emb = np.reshape(emb,(30,512))
            # print(emb)
            name = []
            #for l in range(emb.shape[1]):
                #name.append('x' + str(l))
            df = pd.DataFrame(emb)
            labels = [label] * len(emb)
            #df['labels'] = labels
            #df = df[['labels', *name]]
            fpath = os.path.join(csv_path, f'{label}.csv')
            df.to_csv(fpath, index=False,header=False)

        return map_embeddings




    read_path(path_name)
