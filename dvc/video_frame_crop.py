import cv2
import os
import argparse
import shutil


def get_frame_from_video(video_name, num_s, save_path):
    """
    :param video_name: 输入视频路径
    :param num_s: 保存图片的帧率间隔
    :param save_path: 抽出的图片保存的位置
    """

    # 保存图片的路径
    path = video_name.split('.mp4')[0]
    file_name = path.split('/')[-1]
    print(file_name)

    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    # else:
    #     shutil.rmtree(save_path)
    #     os.makedirs(save_path)
    #     print('path of %s already exist and rebuild' % save_path)

    video_capture = cv2.VideoCapture(video_name)
    fps = video_capture.get(5)
    i = 0
    j = 0

    while True:
        success, frame = video_capture.read()
        i += 1
        if i % int(fps * num_s) == 0:
            # 保存图片
            try:
                j += 1
                save_name = save_path + '/' + file_name + '_' + str(j).zfill(4) + '.jpeg'
                cv2.imwrite(save_name, frame)
            except:
                print('出现未知错误！跳过')

            print('image of %s is saved' % save_name)
        if not success:
            print('video is all read')
            break


if __name__ == '__main__':
    """
    进行视频抽帧处理，根据要求的间隔时间进行图片截取。
    -s (--source): 视频文件路径地址
    -o (--output): 图片文件保存文件夹
    -i (--interval): 多少秒抽一帧
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, required=True, help='mp4 file path')
    parser.add_argument('-o', '--output', type=str, required=True, help='image save path')
    parser.add_argument('-i', '--interval', type=int, default=3, help='interval second')
    opt = parser.parse_args()
    files = os.listdir(opt.source)
    files.sort()
    for file_ in files:
        video_name = opt.source + '/' + file_
        get_frame_from_video(video_name, opt.interval, opt.output)