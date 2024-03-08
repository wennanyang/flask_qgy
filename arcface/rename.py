import os,sys                       #导入模块
def add_prefix_subfolders():        #定义函数名称
    mark = '.jpg'                  #准备添加的前缀内容
    old_names = os.listdir(path0)  #取路径下的文件名，生成列表
    for old_name in old_names:
     i = 1
     path1 = os.listdir(old_name)
     for path in path1:      #遍历列表下的文件名
         if path == ".DS_store":
             pass
         else:
          if path!= sys.argv[0]:     #代码本身文件路径，防止脚本文件放在path路径下时，被一起重命名
             os.rename(os.path.join(path0,old_name,path),os.path.join(path0,old_name,str(i)+mark))  #子文件夹重命名
             print (path," ",str(i)+mark)
             i = i + 1
if __name__ == '__main__':
        path0 = r'/data/arcface-pytorch-main/fengyi_goods/'   #运行程序前，记得修改主文件夹路径！
        if path0 == ".DS_store":
            pass
        else:
         add_prefix_subfolders()            #调用定义的函数

