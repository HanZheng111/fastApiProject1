import time
import uuid

import onnxruntime
import rembg
import torch
from PIL import Image


print(torch.__version__) # pytorch版本
print(torch.version.cuda) # cuda版本
print(torch.cuda.is_available()) # 查看cuda是否可用

print("onnxruntime",onnxruntime.get_device())
print("onnxruntime",onnxruntime.get_available_providers())
# 返回size
def get_resize_widget_number(size, long):
    widget = long * size[1] / size[0]
    return int(long), int(widget)


# 去除背景
def remove_background(img, s):
    oImg = rembg.remove(data=img, session=s)
    return oImg



# ort_session = onnxruntime.InferenceSession('C:\\Users\\hotgame\\.u2net\\u2net.onnx', providers=['CUDAExecutionProvider'])
# # double check is using GPU?
# print(ort_session.get_providers())


rembgSession = rembg.new_session()


def generate_pic(upload_image, raw_image, up_is_top, up_need_remove, upload_image_resize_long,
                  upload_image_resize_width, x, y):
    print(upload_image, raw_image)
    # 都以RGBA模式打开不然后面合成图片有问题
    upload_pic = Image.open(upload_image).convert("RGBA")
    raw_pic = Image.open(raw_image).convert("RGBA")

    # 修改尺寸
    upload_pic = upload_pic.resize((upload_image_resize_long, upload_image_resize_width))


    # 移除背景
    if up_need_remove:
        upload_pic = remove_background(upload_pic, rembgSession)

    if up_is_top:
        result_img = raw_pic
        result_img.paste(upload_pic, (x, y), upload_pic)
    else:
        result_img = Image.new("RGBA", raw_pic.size, (0, 0, 0, 0))
        result_img.paste(upload_pic, (x, y), upload_pic)
        result_img.paste(raw_pic, (0, 0), raw_pic)


    img_path = "output/splicing_pic/" + uuid.uuid4().hex + ".png"
    result_img.save(img_path, format="PNG")

    return img_path


