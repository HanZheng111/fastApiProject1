import netron
import onnx
import onnxruntime
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")
    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

modelData = "../../weights/AnimeGANv3_Hayao_36.onnx"
netron.start(modelData)

onnx_model=onnx.load(modelData)
onnx.checker.check_model(onnx_model)

ort_session=onnxruntime.InferenceSession(modelData, providers=["CPUExecutionProvider"])

# 加载图像与预处理
img = load_image("D:/input/f5125a7c-e278-4f2c-a616-46617fbc5441.png")

resize = transforms.Resize([3, 3])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

# 在ONNX Runtime中运行超分辨率模型
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]

# 从输出张量构造最终输出图像，并保存
img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")  # Cr, Cb通道通过插值发大

final_img.save("D://cat_superres_with_ort.jpg")