import cv2
import torch
import numpy as np
from pandas import *

def yolo_test():
    print(torch.cuda.is_available())
    videoCapture = cv2.VideoCapture('./person/play_phone_5.mp4')


    # img2 = cv2.imread('767.jpg')[..., ::-1]
    # img2 = cv2.resize(img2, (640, 480))

    # write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (1920, 1080)
    # out = cv2.VideoWriter('inference/phone-1.mp4', fourcc, fps, size)

    # model
    primary_model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path='./models/person.pt')
    sercondary_model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path='./models/phone.pt')

    # db
    bb_df = DataFrame(columns=('xmin', 'ymax', 'xmax', 'ymin', 'confidence', 'name'))  # 生成空的pandas表
    i = 0

    success, frame = videoCapture.read()
    frame_count = 0
    while success:
        results = primary_model(frame[..., ::-1])
        points = results.pandas().xyxy[0]
        # print(points)
        bb_df = bb_df.drop(index=bb_df.index)
        i = 0
        for index, row in points.iterrows():
            # 获取两个点的坐标并保存
            if row['name'] == 'person':
                cv2.rectangle(frame, (int(row['xmin']), int(row['ymax'])), (int(row['xmax']), int(row['ymin'])),
                              (0, 255, 0), 2)
                bb_df.loc[index] = [int(row['xmin']), int(row['ymax']), int(row['xmax']), int(row['ymin']),
                                    row['confidence'], row['name']]
                i = i + 1

        # 裁剪图片并检测
        for index, row in bb_df.iterrows():
            # 裁剪图片
            # print(size, int(row['xmin']), int(row['ymax']), int(row['xmax']), int(row['ymin']))
            crop_frame = frame[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]
            # print(index, frame.size, crop_frame.size)
            crop_results = sercondary_model(crop_frame)
            crop_points = crop_results.pandas().xyxy[0]
            if len(crop_points) != 0:
                crop_row = crop_points.iloc[0]
                # print(crop_row['name'], )
                if crop_row['name'] in ['phone']:
                    # print("no_vest", row['confidence'])
                    cv2.rectangle(frame,
                                  (int(row['xmin']) + int(crop_row['xmin']), int(row['ymin']) + int(crop_row['ymax'])),
                                  (int(row['xmin']) + int(crop_row['xmax']), int(row['ymin']) + int(crop_row['ymin'])),
                                  (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame,
                                  (int(row['xmin']) + int(crop_row['xmin']), int(row['ymin']) + int(crop_row['ymax'])),
                                  (int(row['xmin']) + int(crop_row['xmax']), int(row['ymin']) + int(crop_row['ymin'])),
                                  (0, 255, 0), 2)

        frame_count += 1
        print("The number of frame:", frame_count)
        frame = cv2.resize(frame, (1920, 1080))
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
