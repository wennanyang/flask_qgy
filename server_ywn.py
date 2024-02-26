import io
import json
import os
import re
from wsgiref.simple_server import WSGIRequestHandler
import torch
from flask import Flask,jsonify,request
import cv2
from SupContrast.networks.resnet_big import SupConResNet
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pathlib
import pandas as pd
import heapq
import logging
import time
import requests
import json
from tqdm import tqdm

app=Flask(__name__)

def load_model(model_path):
    model=SupConResNet(name="resnet18")
    ckpt=torch.load(model_path, map_location='cpu')
    state_dict = ckpt['model']
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
    cudnn.benchmark = True
    model.load_state_dict(state_dict)
    return model
#transform the opencv image
def transform_image(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image=cv2.imread(image_path)[:,:,::-1]
    dim=(128,128)
    image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    image=torch.from_numpy(image.transpose((2, 0, 1)))
    image=image.float().div(255)
    # image[0]=image[0].sub(0.485).div(0.229)
    # image[1]=image[1].sub(0.456).div(0.224)
    # image[2]=image[2].sub(0.406).div(0.225)
    image[0]=image[0].sub(0).div(1)
    image[1]=image[1].sub(0).div(1)
    image[2]=image[2].sub(0).div(1)
    return image
#load the template
def load_template(template_path):
    csv_list=[]
    template=[]
    for csv_file in pathlib.Path(template_path).glob("*.csv"):
        csv_list.append(str(csv_file))
    csv_list=sorted(csv_list)
    for csv_file in tqdm(csv_list,desc="reading template files", total=len(csv_list), unit="files"):
        df=pd.read_csv(str(csv_file),header=None)
        data=df.values
        data=data.astype(float)
        data=torch.from_numpy(data)
        template.append(data)
    return torch.stack(template,dim=0)
#compare the image to the template

def compare_template_image(template,images,refrigerator_template):
    distances=[100.0 for i in range(template.size()[0])]
    for i in range(images.size()[0]):
        for j in refrigerator_template:
            for k in range(template.size()[1]):
                distances[j]=min(float(F.pairwise_distance(images[i],template[j][k],p=2)),distances[j])
    distances=[-item for item in distances]
    max_number=heapq.nlargest(5,distances)
    max_index=map(distances.index,max_number)
    return list(max_index)

#error get
@app.route('/', methods=['GET'])
def root():
    return jsonify({"msg":"try posting to the /predict "})

#inference
@app.route('/predict',methods=['POST'])
def predict():
    #get the information from POST
    # video_id=request.form.get("order_id")
    # video_address=request.form.get("video_url") 
    data= request.get_json() 
    print("haved accept data")
    video_id = data['order_id']
    video_address = "./video/video.mp4"
    refrigerator_template = data['refigerator_template']
    
    print(f"video_id is {video_id}")
    print(f"video_address is {video_address}")

    #the data(return)
    json_file={}
    outputs=[]
    labels=[]
    frame_info={}
    
    #read the video,yolov5 inference
    cap=cv2.VideoCapture(video_address)
    frame_number=0
    a=time.time()
    while(True):
        ret,img=cap.read()
        if ret:
            img1=img.copy()
            head=[]
            goods=[]
            frame_number+=1
            with torch.no_grad():
                results=model_yolov5_1(img,size=640)
                if results.xyxy[0].size()[0]!=0:
                    for i in range(results.xyxy[0].size()[0]):
                        # 当该矩形框中是商品
                        if results.xyxy[0][i][5]==0:
                            goods.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                            crop_img=img1[int(results.xyxy[0][i][1]):int(results.xyxy[0][i][3]),int(results.xyxy[0][i][0]):int(results.xyxy[0][i][2])]
                            #cv2.imwrite("/home/qgy/flask_product/test_bug/pictures/"+str(frame_number)+'.jpg',crop_img)
                            crop_img1=transform_image(crop_img).unsqueeze(0).to(device_1)
                            outputs.append(model_resnet_1(crop_img1).cpu())
                        # 当该矩形框中是人头
                        elif results.xyxy[0][i][5]==1:
                            head.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                    frame_info[str(frame_number)]={"goods":goods,"head":head}      
        else: 
            break
    print("___________________________________________________________________")
    print(f"elapsed time is {time.time()-a:.3f}s")
    
    if(len(outputs)!=0):
        labels=compare_template_image(template,torch.stack(outputs,0),refrigerator_template)
    else:
        labels='no goods to recommend'

    json_file["frame_number"]=frame_info
    json_file['order_id']=video_id
    json_file['labels']=labels
    
    #post data to a url
    header={
        "Content-Type":"application/json",
        "charset":"UTF-8"
    }
    s = requests.session()
    s.keep_alive = False
    # requests.post("https://api.wrshg.com/datacenter-open-center/test/order/review/discern/result/v1",data=json.dumps(json_file),headers=header)
    return jsonify(json_file)

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    #load the yolov5 model
    device_1=torch.device("cuda",0)
    model_yolov5_1=torch.hub.load('./yolov5',"custom",path="./yolov5/runs/train/exp5/weights/best.pt",source='local')
    model_yolov5_1=model_yolov5_1.to(device_1)
    model_yolov5_1.eval()
    logging.info("____________________load the yolov5 successfully______________")

    #load the resnet model
    model_resnet_1=load_model("./models/ckpt_epoch_1000.pth")
    model_resnet_1.to(device_1)
    model_resnet_1.eval()
    logging.info("____________________load the resnet successfully______________")

    #load the template
    template=load_template("./product_csv/850_template")
    logging.info("____________________load the template successfully____________")


    app.run(host="0.0.0.0",debug=True,port=5001)