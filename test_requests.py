import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
def send_post_request(order_id, video_address, url="http://127.0.0.1:5001/predict"):

    # 定义POST请求的参数
    payload = {
        "order_id": order_id,
        "video_address": video_address
    }
    header={
        "User-Agent": "PostmanRuntime/7.36.3",
        "Content-Type":"application/json",
        "charset":"UTF-8"
    }
    try:
        # 发送POST请求
        response = requests.post(url, json=payload, headers=header, verify=False)

        # 检查响应状态码
        if response.status_code == 200:
            print("POST请求成功!")
            filename = f"./json/responses{time.time():3.1f}.json"
            with open(filename, "w") as f:
                json.dump(response.json(), f, indent=2)
                print(f"writing {filename} sucess!")

        else:
            print(f"POST请求失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"发生异常: {e}")

# 示例调用
if __name__ == '__main__':
    max_worker = 2
    requests_num = 3
    order_id = ["123", "456"] * requests_num
    video_address1 = "https://static.hgobox.com/video/2024-02-26/240226100602013858_main.mp4?e=1708914405&token=JYU7SpuCRem3PGujrkkZIGxZi6gcw5QBrEwwD4G6:xTdc0Yb2RX0aYAxMkog1u5w2GUI="
    video_address2 = "https://static.hgobox.com/video/2024-02-26/240226100624014095_main.mp4?e=1708914429&token=JYU7SpuCRem3PGujrkkZIGxZi6gcw5QBrEwwD4G6:fkvAY37MHrsrFIYIbURC1rni-nc="
    video_address = [video_address1, video_address2] * requests_num

    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        # partial_send_request = functools.partial(send_post_request, param=None)
        executor.map(send_post_request, order_id, video_address)


