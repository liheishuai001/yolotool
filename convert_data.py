import argparse
import os
import xml.etree.ElementTree as ET

import deeplake
import numpy as np

xmlfilepath = './convertlabels'
imagefilepath = './images'
labelpath = './labels'


'''
将按键精灵转换的底层函数
'''
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

'''
读取yolo格式数据
'''
def read_yolo_boxes(label: str):
    box_f = open(label)
    lines = box_f.read()
    box_f.close()
    lines_split = lines.splitlines()
    yolo_boxes = np.zeros((len(lines_split), 4))
    yolo_labels = np.zeros(len(lines_split))
    for l, line in enumerate(lines_split):
        line_split = line.split()
        yolo_boxes[l, :] = np.array(
            (float(line_split[1]), float(line_split[2]), float(line_split[3]), float(line_split[4])))
        yolo_labels[l] = int(line_split[0])

    return yolo_boxes, yolo_labels


def convert_annotation(data: list[str], classes):
    if not os.path.exists(labelpath):
        os.makedirs(labelpath)
    for item in data:
        name = item[:-4]
        infile = open(f'{xmlfilepath}/{name}.xml', encoding='utf-8')
        outfile = open(f'{labelpath}/{name}.txt', 'w', encoding='utf-8')
        tree = ET.parse(infile)
        root = tree.getroot()
        size = root.find('size')
        if size != None:
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                # 如果类别不是对应在我们预定好的class文件中，或difficult==1则跳过
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                # 带入进行归一化操作
                # w = 宽, h = 高， b= bndbox的数组 = ['xmin','xmax','ymin','ymax']
                bb = convert((w, h), b)
                outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    print("convert success total: %s", len(data))


'''
实现数据上报数据版本控制
'''
def upload_data(data, classes, args):
    ds = deeplake.load(f's3://{args.bucket}', creds={
        'aws_access_key_id': args.access_key,
        'aws_secret_access_key': args.secret_key,
        'endpoint_url': args.url
    })
    with ds:
        for item in data:
            name = data[item][:-4]
            yolo_boxes, yolo_labels = read_yolo_boxes(os.path.join(xmlfilepath, name))
            imagefilename = os.path.join(imagefilepath, name + '.png') if os.path.exists(os.path.join(imagefilepath, name + '.png')) else os.path.join(imagefilepath, name + '.jpg')
            ds.append({'images': deeplake.read(imagefilename),
                       'labels': yolo_labels.astype(np.uint32),
                       'boxes': yolo_boxes.astype(np.float32)
                       })
    print('upload success total: %s', len(data))


'''
将按键精力导出的标注数据转换为yolo可识别的结论，需要在脚本运行目录下新建/convert/labels存放需要转换的标注
数据，最终完成转换的标注将在当前目录的labels下呈现，为对应具体的分类需要提供classes并按顺序提供。
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classes', type=str, nargs='+', required=True, help='class name')
    parser.add_argument('-u', '--upload', action='store_false', default=False, help='auto update dvc')
    opt = parser.parse_args()
    total_xml = os.listdir(xmlfilepath)
    if len(total_xml) <= 0:
        raise IndexError('must have one labels')
    convert_annotation(total_xml, opt.classes)
    if opt.upload:
        upload_data(total_xml, opt.classes, opt)
    exit()
