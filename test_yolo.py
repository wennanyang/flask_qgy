import torch
import cv2
import os
yolo_model_path = "./models/best.pt"
video_adress = "https://static.hgobox.com/video/2024-02-26/240226100602013858_main.mp4?e=1708914405&token=JYU7SpuCRem3PGujrkkZIGxZi6gcw5QBrEwwD4G6:xTdc0Yb2RX0aYAxMkog1u5w2GUI="
save_dir = "e:/image/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
device = torch.device("cuda", 0)
model_yolov5 = torch.hub.load('./yolov5',"custom",path=yolo_model_path,source='local')
model_yolov5 = model_yolov5.to(device=device)
model_yolov5.eval()


cap=cv2.VideoCapture(video_adress)
frame_num = 0
while (True) :
    ret, img = cap.read()
    if ret :
        img_copy = img.copy()
        frame_num += 1
        with torch.no_grad():
            results=model_yolov5(img,size=640)
            if results.xyxy[0].size()[0]!=0:

                for i in range(results.xyxy[0].size()[0]):
                    if results.xyxy[0][i][5]==0:
                        x1, y1, x2, y2 = int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])
                        crop_img=img_copy[x1:x2, y1:y2]
                        cv2.imwrite(os.path.join(save_dir, f"goods_{frame_num}.jpg"), crop_img)
                    elif results.xyxy[0][i][5]==1:
                        x3, y3, x4, y4 = int(results.xyxy[0][i][0]),int(results.xyxy[0][i][1]),int(results.xyxy[0][i][2]),int(results.xyxy[0][i][3])
                        crop_img_head = img_copy[x3:x4, y3:y4]
                        cv2.imwrite(os.path.join(save_dir, f"head_{frame_num}.jpg"), crop_img_head)
    else :
        break
