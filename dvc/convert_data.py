import argparse
import os
import xml.etree.ElementTree as ET
import deeplake
import numpy as np


def convert(size, box):
    """
    将按键精灵转换的底层函数
    """
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


def read_yolo_boxes(label: str):
    """
    读取yolo格式数据
    """
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


def convert_annotation(data: list[str], convert_path, labels_path, classes):
    """
    遍历xml标注数据进行转换，并写入到yolo格式中
    """
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    for item in data:
        name = item[:-4]
        infile = open(f'{convert_path}/{name}.xml', encoding='utf-8')
        outfile = open(f'{labels_path}/{name}.txt', 'w', encoding='utf-8')
        tree = ET.parse(infile)
        root = tree.getroot()
        size = root.find('size')
        if size is not None:
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
    print("convert success total: %s" % len(data))


def upload_data(data, labels_path, images_path, bucket, access_key, secret_key,
                url):
    """
    实现数据上报数据版本控制
    """
    ds = deeplake.load(f's3://{bucket}', creds={
        'aws_access_key_id': access_key,
        'aws_secret_access_key': secret_key,
        'endpoint_url': url
    })
    with ds:
        for item in data:
            name = data[item][:-4]
            yolo_boxes, yolo_labels = read_yolo_boxes(os.path.join(labels_path, name + '.txt'))
            imagefilename = os.path.join(images_path, name + '.png') if os.path.exists(os.path.join(images_path, name + '.png')) else os.path.join(images_path, name + '.jpg')
            ds.append({'images': deeplake.read(imagefilename),
                       'labels': yolo_labels.astype(np.uint32),
                       'boxes': yolo_boxes.astype(np.float32)
                       })
    print('upload success total: %s' % len(data))


if __name__ == '__main__':
    """
    实现从精灵标注助手进行数据转换，并将转换后的数据上传至deeplake数据仓库中从而实现数据版本控制，为此我们需要按照
    如下的方式使用该脚本。
    -s (--source): 需要进行转换的数据源文件夹路径
    -o (--output): 转换后标注数据的存储文件夹路径
    -img (--image_path): 需上传的图片文件夹路径
    -b (--bucket): 数据仓库名称，需要保障每个仓库的名称唯一
    -ak (--access_key): S3接口访问令牌
    -sk (--secret_key): S3接口访问秘钥
    -u (--url): S3访问地址
    -c (--classes): 上报数据的分类名称，可支持多个
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, default='../convertlabels', help='convert annotations path')
    parser.add_argument('-o', '--output', type=str, default='../labels', help='yolo annotations path')
    parser.add_argument('-img', '--image_path', type=str, default='../images', help='yolo image path')
    parser.add_argument('-b', '--bucket', type=str, default='', help='s3 bucket name')
    parser.add_argument('-ak', '--access_key', type=str, help='s3 access key')
    parser.add_argument('-sk', '--secret_key', type=str, help='s3 access secret key')
    parser.add_argument('-u', '--url', type=str, default='', help='s3 endpoint url')
    parser.add_argument('-c', '--classes', type=str, nargs='+', help='model class name')
    opt = parser.parse_args()
    total_xml = os.listdir(opt.source)
    if len(total_xml) <= 0:
        raise IndexError('must have one labels')
    convert_annotation(total_xml, opt.source, opt.output, opt.classes)
    if len(opt.bucket.strip()) > 0:
        upload_data(total_xml, opt.output, opt.image_path, opt.bucket, opt.access_key, opt.secret_key, opt.url)
    exit()
