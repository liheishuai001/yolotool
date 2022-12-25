import cv2
import torch
import numpy as np
from pandas import *


def yolo_test():
    print(torch.cuda.is_available())
    videoCapture = cv2.VideoCapture('../person/no_hat_1.mp4')

    # img2 = cv2.imread('767.jpg')[..., ::-1]
    # img2 = cv2.resize(img2, (640, 480))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (1920, 1080)
    # out = cv2.VideoWriter('./inference/phone-1.mp4', fourcc, fps, size)

    # model
    primary_model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path='../models/helmet.pt')

    success, frame = videoCapture.read()
    while success:
        results = primary_model(frame[..., ::-1])
        points = results.pandas().xyxy[0]
        if len(points) != 0:
            # print(crop_row['name'], )
            for index, row in points.iterrows():
                if row['name'] == 'head':
                    # print("no_vest", row['confidence'])
                    cv2.rectangle(frame,
                              (int(row['xmin']), int(row['ymax'])),
                              (int(row['xmax']), int(row['ymin'])),
                              (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame,
                              (int(row['xmin']), int(row['ymax'])),
                               (int(row['xmax']), int(row['ymin'])),
                              (0, 255, 0), 2)

        # frame = cv2.resize(frame, (1920, 1080))
        # out.write(frame)
        cv2.imshow('video_mp4', frame)
        # cv2.moveWindow('video_mp4', 500, 0)

        success, frame = videoCapture.read()  # 获取下一帧

        if cv2.waitKey(1) == ord('q'):
            break

    videoCapture.release()
    # out.release()
    cv2.destroyAllWindows()
    print("Finishied!")


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    yolo_test()
