import datetime
import os

from api.endpoints import getFace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch.nn.functional as F
import uuid


# -------------------------- hy add 01 --------------------------
class ConvNormLReLU(torch.nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero": torch.nn.ZeroPad2d,
            "same": torch.nn.ReplicationPad2d,
            "reflect": torch.nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            torch.nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            torch.nn.LeakyReLU(0.2, inplace=True)
        )


class InvertedResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(torch.nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(torch.nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out


class Generator(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.block_a = torch.nn.Sequential(
            ConvNormLReLU(3, 32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(64, 64)
        )

        self.block_b = torch.nn.Sequential(
            ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            ConvNormLReLU(128, 128)
        )

        self.block_c = torch.nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )

        self.block_d = torch.nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = torch.nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64, 64),
            ConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out


# -------------------------- hy add 02 --------------------------

def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")
    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32

        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def generate(image_path: str, output_dir: str, t: int, device='cuda'):
    # 加载图片
    image = load_image(image_path)

    # 图片脸部识别
    image = getFace.crop_one_face(image)

    # 图片重设
    size = image.size
    print(size)
    y = 512
    x = int(size[0] * y / size[1])
    image = image.resize(size=(x, y))

    print(image_path, "开始加载", datetime.datetime.now())
    _ext = os.path.basename(image_path).strip().split('.')[-1]
    if t == 1:
        _checkpoint = 'weights/paprika.pt'
    elif t == 2:
        _checkpoint = 'weights/face_paint_512_v2.pt'
    elif t == 3:
        _checkpoint = 'weights/face_paint_512_v1.pt'
    # elif t == 4:
    #     _checkpoint = 'weights/vgg19_no_fc.npy'
    elif t == 5:
        _checkpoint = 'weights/pytorch_generator_Shinkai.pt'
    elif t == 6:
        _checkpoint = 'weights/Hayao.pt'
    else:
        raise Exception('type not support')
    os.makedirs(output_dir, exist_ok=True)
    net = Generator()
    net.load_state_dict(torch.load(_checkpoint, weights_only=True))
    net.to(device).eval()

    print(image_path, '加载完毕', datetime.datetime.now())
    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = net(image.to(device), False).cuda()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    print(image_path, '运算完毕', datetime.datetime.now())
    # result = os.path.join(output_dir, '/{}.{}'.format(uuid.uuid1().hex, _ext))

    file_path = output_dir + "/" + uuid.uuid4().hex + "." + _ext
    print(file_path)
    out.save(file_path)
    return file_path
#
# if __name__ == '__main__':
#     # print(handle('samples/images/fengjing.jpg', 'samples/images_result/', 1))
#     # print(handle('samples/images/renxiang.jpg', 'samples/images_result/', 2))
#     # print(handle('D:/test_input/OIP.jpg', 'D:/test_output/', 2))
#     # print(handle('D:/test_input/OIP (1).jpg', 'D:/test_output/', 2))
#     # print(handle('D:/test_input/R.jpg', 'D:/test_output/', 2))
#     # print(handle('D:/test_input/222.jpg', 'D:/test_output/', 2))
#     # print(handle('D:/test_input/3333.jpg', 'D:/test_output/', 2))
#
#     print(handle('D:/test_input/OIP (1).jpg', 'D:/cartoon_output/', 2))
#     print(handle('D:/test_input/OIP (1).jpg', 'D:/cartoon_output/', 2))
#     print(handle('D:/test_input/OIP (1).jpg', 'D:/cartoon_output/', 2))
#     print(handle('D:/test_input/OIP (1).jpg', 'D:/cartoon_output/', 2))
