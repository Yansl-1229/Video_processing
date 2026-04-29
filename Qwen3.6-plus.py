import os
from dashscope import MultiModalConversation
import dashscope 

# 各地域配置不同，请根据实际地域修改
dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

local_path1 = "output/segments/04/frames_ppt/frame_001500000.png"
local_path2 = "output/segments/04/frames_ppt/frame_001526000.png"
local_path3 = "output/segments/04/frames_ppt/frame_001710000.png"
local_path4 = "output/segments/04/frames_ppt/frame_001912000.png"
local_path5 = "output/segments/04/frames_ppt/frame_001972000.png"
local_path6 = "output/segments/04/frames_ppt/frame_002144000.png"

image_path1 = f"file://{local_path1}"
image_path2 = f"file://{local_path2}"
image_path3 = f"file://{local_path3}"
image_path4 = f"file://{local_path4}"
image_path5 = f"file://{local_path5}"
image_path6 = f"file://{local_path6}"

messages = [{'role':'user',
              #  传入图像列表时，fps 参数适用于Qwen3.6、Qwen3-VL 和 Qwen2.5-VL系列模型
             'content': [{'video': [image_path1,image_path2,image_path3,image_path4,image_path5,image_path6]},
                            {'text': '这些图片描绘的是什么景象?'}]}]
response = MultiModalConversation.call(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen3.6-plus',  
    messages=messages)
print(response.output.choices[0].message.content[0]["text"])