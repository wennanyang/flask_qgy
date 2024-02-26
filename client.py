import requests
from flask import jsonify
import json

header={
        "Content-Type":"application/json",
        "charset":"UTF-8"
    }


data1={"order_id":"12345678","video_url":"https://static.hgobox.com/video/2022-07-21/220721151502076709_0.mp4","refigerator_template":[1,2,3,4]}
data2={"order_id":"12345678","video_url":"https://static.hgobox.com/video/2022-07-20/220720142155029007_main.mp4"}
#data1={'order_id':'2345678','video_url':'https://static.hgobox.com/video/2022-07-06/220706100708086070_main.mp4'}

one=requests.post("http://localhost:5001/predict",data=json.dumps(data1),headers=header)
two=requests.post("http://localhost:5001/predict",data=json.dumps(data2),headers=header)
print(one.json())

print(two.json())
