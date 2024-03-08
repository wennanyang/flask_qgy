import io
import json
import os
import re
from wsgiref.simple_server import WSGIRequestHandler
import torch
from flask import Flask,jsonify,request
import cv2

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pathlib
import pandas as pd
import heapq
import logging
import time
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm 
from arcface.nets.arcface import Arcface
from arcface.utils.utils import get_num_classes
app=Flask(__name__)
executor=ThreadPoolExecutor(5)
#CUDA
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"
gpu_log=[0, 0, 0, 0, 0]

#determine use which gpu
def detect_gpu():
    for i in range(len(gpu_log)):
        if gpu_log[i]==0:
            gpu_log[i]=1
            return i
        else:
            continue
    return -1

def load_yolov5_and_arcface_model(gpu_index, yolo_model_path, arcface_model_path):
    device = torch.device("cuda", gpu_index)

    model_yolov5 = torch.hub.load('./yolov5',"custom",path=yolo_model_path,source='local')
    model_yolov5 = model_yolov5.to(device=device)
    model_yolov5.eval()

    annotation_path = "arcface/r553.txt"
    num_classes = get_num_classes(annotation_path=annotation_path)
    backbone = "resnet18"
    pretrained = False
    arcface_model = Arcface(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    ckpt=torch.load(arcface_model_path, map_location='cpu')
    # state_dict = ckpt['model']
    # if torch.cuda.is_available():
    #     if torch.cuda.device_count() > 1:
    #         arcface_model.encoder = torch.nn.DataParallel(arcface_model.encoder)
    #     else:
    #         new_state_dict = {}
    #         for k, v in state_dict.items():
    #             k = k.replace("module.", "")
    #             new_state_dict[k] = v
    #         state_dict = new_state_dict
    arcface_model = arcface_model.to(device=device)
    cudnn.benchmark = True
    arcface_model.load_state_dict(state_dict=ckpt)
    arcface_model.eval()
    return model_yolov5, arcface_model



#transform the opencv image
def transform_image(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #image=cv2.imread(image_path)[:,:,::-1]
    dim=(128,128)
    image=cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
    image=torch.from_numpy(image.transpose((2, 0, 1)))
    image=image.float().div(255)
    image[0]=image[0].sub(0.485).div(0.229)
    image[1]=image[1].sub(0.456).div(0.224)
    image[2]=image[2].sub(0.406).div(0.225)
    return image

#load the template
def load_template(template_path):
    template=[]
    for csv_file in tqdm(pathlib.Path(template_path).glob("*.csv"), desc="reading template", unit="files"):
        df=pd.read_csv(str(csv_file),header=None)
        data=df.values
        data=data.astype(float)
        data=torch.from_numpy(data)
        template.append(data)
    return torch.stack(template,dim=0)


#compare the image to the template
def compare_template_image(template,images):
    distances=[100.0 for i in range(len(template))]
    for i in range(images.size()[0]):
        for j in range(template.size()[0]):
            for k in range(template.size()[1]):
                distances[j]=min(float(F.pairwise_distance(images[i],template[j][k],p=2)),distances[j])
    distances=[-item for item in distances]
    max_number=heapq.nlargest(5,distances)
    max_index=map(distances.index,max_number)
    return list(max_index)

#error get
@app.route('/',methods=['GET'])
def root():
    return jsonify({"msg":"try posting to the /predict "})

#inference
@app.route('/predict',methods=['POST'])
def run_app():
    data= request.get_json()
    video_id = data['order_id']
    video_address = data['video_address']
    # print(data)
    a=executor.submit(predict,video_id,video_address)
    return jsonify(a.result())
    



def predict(arg1,arg2):     
    json_file={}
    outputs=[]
    labels=[]
    frame_info={}
    
    #detect gpu
    gpu_detect=detect_gpu()
    if gpu_detect != -1:
        logging.info(f"using gpu{gpu_detect}")

    #read the video,yolov5 inference
    cap=cv2.VideoCapture(arg2)
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
                if gpu_detect != -1:
                    model_yolov5, model_arcface = yolov5_arcface_list[gpu_detect]
                    results=model_yolov5(img,size=640)
                    if results.xyxy[0].size()[0]!=0:
                        for i in range(results.xyxy[0].size()[0]):
                            if results.xyxy[0][i][5]==0:
                                goods.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                                crop_img=img1[int(results.xyxy[0][i][1]):int(results.xyxy[0][i][3]),int(results.xyxy[0][i][0]):int(results.xyxy[0][i][2])]
                                crop_img1=transform_image(crop_img).unsqueeze(0).to(torch.device("cuda", gpu_detect))
                                outputs.append(model_arcface(crop_img1).cpu())
                            elif results.xyxy[0][i][5]==1:
                                head.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                        frame_info[str(frame_number)]={"goods":goods,"head":head}
                # if gpu_detect == 0:
                #     results=model_yolov5_0(img,size=640)
                #     if results.xyxy[0].size()[0]!=0:
                #         for i in range(results.xyxy[0].size()[0]):
                #             if results.xyxy[0][i][5]==0:
                #                 goods.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #                 crop_img=img1[int(results.xyxy[0][i][1]):int(results.xyxy[0][i][3]),int(results.xyxy[0][i][0]):int(results.xyxy[0][i][2])]
                #                 crop_img1=transform_image(crop_img).unsqueeze(0).to(torch.device("cuda", gpu_detect))
                #                 outputs.append(model_arcface_0(crop_img1).cpu())
                #             elif results.xyxy[0][i][5]==1:
                #                 head.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #         frame_info[str(frame_number)]={"goods":goods,"head":head}
                # elif gpu_detect == 1:
                #     results=model_yolov5_1(img,size=640)
                #     if results.xyxy[0].size()[0]!=0:
                #         for i in range(results.xyxy[0].size()[0]):
                #             if results.xyxy[0][i][5]==0:
                #                 goods.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #                 crop_img=img1[int(results.xyxy[0][i][1]):int(results.xyxy[0][i][3]),int(results.xyxy[0][i][0]):int(results.xyxy[0][i][2])]
                #                 crop_img1=transform_image(crop_img).unsqueeze(0).to(torch.device("cuda", gpu_detect))
                #                 outputs.append(model_arcface_1(crop_img1).cpu())
                #             elif results.xyxy[0][i][5]==1:
                #                 head.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #         frame_info[str(frame_number)]={"goods":goods,"head":head}
                # elif gpu_detect == 2:
                #     results=model_yolov5_2(img,size=640)
                #     if results.xyxy[0].size()[0]!=0:
                #         for i in range(results.xyxy[0].size()[0]):
                #             if results.xyxy[0][i][5]==0:
                #                 goods.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #                 crop_img=img1[int(results.xyxy[0][i][1]):int(results.xyxy[0][i][3]),int(results.xyxy[0][i][0]):int(results.xyxy[0][i][2])]
                #                 crop_img1=transform_image(crop_img).unsqueeze(0).to(torch.device("cuda", gpu_detect))
                #                 outputs.append(model_arcface_2(crop_img1).cpu())
                #             elif results.xyxy[0][i][5]==1:
                #                 head.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #         frame_info[str(frame_number)]={"goods":goods,"head":head}
                # elif gpu_detect == 3:
                #     results=model_yolov5_3(img,size=640)
                #     if results.xyxy[0].size()[0]!=0:
                #         for i in range(results.xyxy[0].size()[0]):
                #             if results.xyxy[0][i][5]==0:
                #                 goods.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #                 crop_img=img1[int(results.xyxy[0][i][1]):int(results.xyxy[0][i][3]),int(results.xyxy[0][i][0]):int(results.xyxy[0][i][2])]
                #                 crop_img1=transform_image(crop_img).unsqueeze(0).to(torch.device("cuda", gpu_detect))
                #                 outputs.append(model_arcface_3(crop_img1).cpu())
                #             elif results.xyxy[0][i][5]==1:
                #                 head.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])  
                #         frame_info[str(frame_number)]={"goods":goods,"head":head}
                # elif gpu_detect == 4:
                #     results=model_yolov5_4(img,size=640)
                #     if results.xyxy[0].size()[0]!=0:
                #         for i in range(results.xyxy[0].size()[0]):
                #             if results.xyxy[0][i][5]==0:
                #                 goods.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])
                #                 crop_img=img1[int(results.xyxy[0][i][1]):int(results.xyxy[0][i][3]),int(results.xyxy[0][i][0]):int(results.xyxy[0][i][2])]
                #                 crop_img1=transform_image(crop_img).unsqueeze(0).to(torch.device("cuda", gpu_detect))
                #                 outputs.append(model_arcface_4(crop_img1).cpu())
                #             elif results.xyxy[0][i][5]==1:
                #                 head.append([int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])])  
                #         frame_info[str(frame_number)]={"goods":goods,"head":head}
                else:
                    logging.error("no useful gpu")
                
        else: 
            break
    print("_________________________________________________________________________")
    print(f"elapsed time : \t{time.time()-a:.3f}s")
    if len(outputs) != 0:
        labels=compare_template_image(template,torch.stack(outputs,0))
    else:
        labels = ["no goods"]

    json_file["frame_number"]=frame_info
    json_file['order_id']=arg1
    json_file['labels']=labels
    
    #post data to a url
    header={
        "Content-Type":"application/json",
        "charset":"UTF-8"
    }
    # requests.post("https://api.wrshg.com/datacenter-open-center/test/order/review/discern/result/v1",data=json.dumps(json_file),headers=header)
    gpu_log[gpu_detect]=0
    return json_file

if __name__=="__main__":
    yolov5_weight_path = "./models/best.pt"
    arcface_weight_path = "./models/arcface.pth"
    model_yolov5_0, model_arcface_0 = load_yolov5_and_arcface_model(0, yolov5_weight_path, arcface_weight_path)
    model_yolov5_1, model_arcface_1 = load_yolov5_and_arcface_model(1, yolov5_weight_path, arcface_weight_path)
    model_yolov5_2, model_arcface_2 = load_yolov5_and_arcface_model(2, yolov5_weight_path, arcface_weight_path)
    model_yolov5_3, model_arcface_3 = load_yolov5_and_arcface_model(3, yolov5_weight_path, arcface_weight_path)
    model_yolov5_4, model_arcface_4 = load_yolov5_and_arcface_model(4, yolov5_weight_path, arcface_weight_path)
    yolov5_arcface_list = [[model_yolov5_0, model_arcface_0],[model_yolov5_1, model_arcface_1],[model_yolov5_2, model_arcface_2],
                  [model_yolov5_3, model_arcface_3],[model_yolov5_4, model_arcface_4]]
    logging.info("__________________load the yolov5 and arcface successfully____________")

    template=load_template("./arcface/csv")
    logging.info("__________________load the template successfully______________________")

    app.run(host="0.0.0.0",debug=True,port=5001)