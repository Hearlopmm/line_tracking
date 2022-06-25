import cv2
import numpy as np


def cv_show(name, img):  # 定义一个函数，显示图片
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cap = cv2.VideoCapture('D:/aaa/line.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧数
if cap.isOpened():
    ret, img = cap.read()
    while ret:
        cright = 0
        cleft = 0  # 发送给车左右轮的控制量
        th = 0
        r = 0
        i = 0
        p = 0
        prrho = 0
        prtheta = 0
        l = img.shape[1]
        h = img.shape[0]
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化灰度图
        retval, dst = cv2.threshold(imggray, 100, 255, cv2.THRESH_TOZERO_INV)  # 二值化
        retval1, dst1 = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY_INV)  # 二值化
        mask = cv2.erode(dst1, None, iterations=5)  # 腐蚀
        mask = cv2.dilate(mask, None, iterations=2)  # 膨胀
        # cv_show('',mask)
        imgcanny = cv2.Canny(mask, 50, 150)  # canny轮廓检测
        line_points = cv2.HoughLines(imgcanny, 1, np.pi / 180, 130)  # 二维数组line_points储存角度、距离（1像素1角度）
        if line_points is not None:
            for line in line_points:
                rho, theta = line[0]
                r = r + rho
                th = th + theta
                if abs(prrho - rho) < 50:
                    continue
                i = i + 1
                if i == 2:
                    break
                prrho = rho
                prtheta = theta
            th = th / i
            color = mask[300]
            bl_count = np.sum(color == 0)
            bl_index = np.where(color == 0)
            center = (bl_index[0][bl_count - 1] + bl_index[0][0]) / 2
            ramp = center - l / 2
            d = round(ramp * abs(np.cos(th)), 2)
            angle = round(th * 180 / np.pi, 3)
            if 10 > angle or angle > 160:
                print('角度:', angle, '距离:', d)
                '''
                local_line:  left==d<0;  right==d>0
                ramp_line:   /==angle<90; \==angle>90
                '''
                # 发挥部分 只用了p
                if angle < 90 and d > 0:
                    p = 0.5 * d / 300 + 0.5 * angle / 10
                    cright = 100 * (1 - 0.6 * p)
                    cleft = 100 * (1 + 0.6 * p)
                if angle > 90 and d < 0:
                    angle = 180 - angle
                    d = -d
                    p = 0.5 * d / 300 + 0.5 * angle / 20
                    cright = 100 * (1 + 0.6 * p)
                    cleft = 100 * (1 - 0.6 * p)
                if angle < 90 and d < 0:
                    d = -d
                    p = 0.8 * d / 300 - 0.2 * angle / 10
                    cleft = 100 * (1 - 0.5 * p)
                    cright = 100  # 小误差情况，只变一轮
                if angle > 90 and d > 0:
                    angle = 180 - angle
                    p = 0.8 * d / 300 - 0.2 * angle / 10
                    cright = 100 * (1 - 0.5 * p)
                    cleft = 100  # 小误差情况，只变一轮
                print("右轮改变:", round(cright, 2), "%;左轮变为:", round(cleft, 2), "%")
        cv2.waitKey(int(1500 / fps))  # 延时
        cv2.imshow('windows', img)
        ret, img = cap.read()

cap.release()
