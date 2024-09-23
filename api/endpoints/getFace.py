import numpy as np
import cv2
import os

from PIL import Image
from keras.src.losses import hinge


def crop_face(input_folder_path, output_folder_path):
    # 加载面部识别模型
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    images = os.listdir(input_folder_path)
    for image in images:
        image_path = os.path.join(input_folder_path, image)
        img = cv2.imread(image_path)
        height, width, channels = img.shape
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 检测面部
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # 无法识别面部的图片
        if len(faces) == 0:
            print(f"No face found in {image_path}")
            continue

        if len(faces) > 0:
            # 取第一个脸部位置，这里假设一张图片只有一个脸部特征
            # x、y 为人脸的像素位置，w、h 为人脸的宽度和高度。
            x, y, w, h = faces[0]
            # 确定最大正方形的位置
            # 原图片竖方向长，截取正方形长度为原图横方向长，square_size截取正方形的长度
            if height > width:
                square_size = width
                x1 = 0
                x2 = square_size
                # 原图面部靠上
                if y < square_size / 2:
                    y1 = 0
                    y2 = square_size
                # 原图面部靠下
                else:
                    y1 = int(square_size / 2)
                    y2 = height
            # 原图片是横方向长，截取正方形长度为原图竖方向长度
            else:
                square_size = height
                y1 = 0
                y2 = square_size
                # 原图面部靠右
                if x < square_size / 2:
                    x1 = 0
                    x2 = square_size
                # 原图面部靠左
                else:
                    x1 = int(square_size / 2)
                    x2 = square_size

            # 根据最大正方形位置裁剪图片并保存
            cropped_img = img[y1:y2, x1:x2]
            # 调整图像大小为512x512
            resized = cv2.resize(cropped_img, (512, 512), interpolation=cv2.INTER_AREA)
            output_path = os.path.join(output_folder_path, image)
            cv2.imwrite(output_path, resized)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def crop_one_face(in_img):
    # face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 图片格式转换 PIL Image => cv Image
    img = cv2.cvtColor(np.asarray(in_img), cv2.COLOR_RGB2BGR)
    height, width, channels = img.shape
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 检测面部
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # 无法识别面部的图片
    if len(faces) == 0:
        print("no face")
        # print(f"No face found in {image_path}")
        return in_img

    if len(faces) > 0:
        # 取第一个脸部位置，这里假设一张图片只有一个脸部特征
        # x、y 为人脸的像素位置，w、h 为人脸的宽度和高度。
        x, y, w, h = faces[0]

        x1=x
        x2=x+w
        # 拓宽横向
        x1_left=x1
        x2_right=width-x2
        x1_x2=x2-x1

        # x_margin=min(x1_left,x2_right,x1_x2)
        # x1=x1-x_margin
        # x2=x2+x_margin


        y1=y
        y2=y+h
        # 拓宽横向
        y1_left = y1
        y2_right = height - y2
        y1_y2 = y2 - y1

        # y_margin = min(y1_left, y2_right, y1_y2)
        # y1 = y1 - y_margin
        # y2 = y2 + y_margin

        print(  x, y, w, h )
        print(x1,x2,y1,y2)
        print(x1_left,x2_right,x1_x2,y1_left, y2_right, y1_y2)

        margin=min(x1_left,x2_right,x1_x2,y1_left, y2_right, y1_y2)
        x1=x1-margin
        x2=x2+margin
        y1 = y1 - margin
        y2 = y2 + margin

        # # 确定最大正方形的位置
        # # 原图片竖方向长，截取正方形长度为原图横方向长，square_size截取正方形的长度
        # if height > width:
        #     square_size = width
        #     x1 = 0
        #     x2 = square_size
        #     # 原图面部靠上
        #     if y < square_size / 2:
        #         y1 = 0
        #         y2 = square_size
        #     # 原图面部靠下
        #     else:
        #         y1 = int(square_size / 2)
        #         y2 = height
        # # 原图片是横方向长，截取正方形长度为原图竖方向长度
        # else:
        #     square_size = height
        #     y1 = 0
        #     y2 = square_size
        #     # 原图面部靠右
        #     if x < square_size / 2:
        #         x1 = 0
        #         x2 = square_size
        #     # 原图面部靠左
        #     else:
        #         x1 = int(square_size / 2)
        #         x2 = square_size

        # 根据最大正方形位置裁剪图片并保存
        cropped_img = img[y1:y2, x1:x2]
        # print(y1,y2,x1,x2)
        # # 调整图像大小为512x512
        # resized = cv2.resize(cropped_img, (256,256), interpolation=cv2.INTER_AREA)
        # image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        return image


if __name__ == "__main__":
    input_folder = "D:\input"
    output_folder = "D:\output"
    # 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    crop_face(input_folder, output_folder)
    print('Done!')