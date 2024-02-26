import os

folder_path = '/data/codes/flask_product/product_csv/features/'  #文件路径
num = 1

if __name__ == '__main__':
    file_list = os.listdir(folder_path)
    file_list.sort()
    for file in range(0,len(file_list)):
        s = '%04d' % num  # 04表示0001,0002等命名排序
        os.rename(os.path.join(folder_path, str(file)+'.csv'), os.path.join(folder_path, str(s) + '.csv')) #图片格式
        num += 1