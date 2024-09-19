import cv2

def get_face(path):
    img=cv2.imread(path)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("cascades\\haarcascade_frontalface_alt2.xml")

    faces=face_cascade.detectMultiScale(gray,1.3,5)
    print('Number of detected faces:', len(faces))

    if len(faces)>0:
        for i, (x, y, w, h) in enumerate(faces):
            # 在脸部绘制矩形
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = img[y:y + h, x:x + w]
            cv2.imshow("Cropped Face", face)
            cv2.imwrite(f'face{i}.jpg', face)
            print(f"face{i}.jpg is saved")

    # 显示带有检测到的人脸的图像
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()