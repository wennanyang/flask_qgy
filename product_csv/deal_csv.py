import pathlib
import pandas as pd
#删除所有csv的特定行和列
# def standard(root):
#     for i in pathlib.Path(root).rglob("*.csv"):
#         df=pd.read_csv(str(i))
#         df.drop("labels",axis=1,inplace=True)
#         df.drop(df.columns[0:2],axis=1,inplace=True)
        
#         df.to_csv(str(i.parent)+"\\"+str(i.stem).zfill(3)+".csv",index=None)

def  convert_csv(root,target):
    for item in pathlib.Path(root).rglob("*.csv"):
        df=pd.read_csv(str(item))
        #df.head(30).to_csv(str(pathlib.Path(target,item.name)))
        a=[]
        for i in range(0,len(df)):
            if(i>29):
                break
            a.append(i)

        file=df.iloc[a]
        f=pd.DataFrame(file)
        f.drop("labels",axis=1,inplace=True)

        f.to_csv(str(pathlib.Path(target,str(item.stem).zfill(4)+".csv")),header=None,index=None)




if __name__=="__main__":
    # standard(r"C:\Users\sunfei\Desktop\temp_files\class_features_30")
    convert_csv(r'/home/qgy/flask_product/product_csv/features/',r'/home/qgy/flask_product/product_csv/850_template/')

