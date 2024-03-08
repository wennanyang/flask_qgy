import pathlib
import pandas as pd
def generate_csv(root,csv_path):
    mylist=[]
    for i in pathlib.Path(root).rglob("*.jpg"):
        print(str(i),str(i.parent.name))
        mylist.append({"paths":str(i),"labels":str(i.parent.name)})
    df=pd.DataFrame(mylist,columns=["paths","labels"])
    df.to_csv(csv_path,index=False)
if __name__=="__main__":
    generate_csv("/data/arcface-pytorch-main/test", "test.txt")
