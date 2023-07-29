import argparse
import fiftyone as fo
from fiftyone import ViewField as F
import torch
import numpy as np
from yolo.hubconf import custom
from yolo.utils.general import cv2


def xyxytoxywh(bbox, size):
    x = bbox[0] / size[0]
    y = bbox[1] / size[1]
    w = (bbox[2] - bbox[0]) / size[0]
    h = (bbox[3] - bbox[1]) / size[1]
    return x, y, w, h


def predictions(view, dataset, model_path):
    """
    进行推理，实现将yolo格式标注数据与yolo推理结论进行可视化
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes = dataset.default_classes
    # model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', model_path)
    model = custom(model_path)
    model.conf = 0.1
    model.eval()

    with fo.ProgressBar() as pb:
        for sample in pb(view):
            detections = []
            #raw_data = np.fromfile(sample.filepath, dtype=np.uint8)
            image = cv2.imread(sample.filepath)
            if not hasattr(image, 'shape'):
                continue
            h, w, *c = image.shape
            results = model(image[:, :, ::-1], size=512)
            points = results.pandas().xyxy[0]
            if len(points) != 0:
                for index, row in points.iterrows():
                    label = row['name']
                    score = row['confidence']
                    box = xyxytoxywh([int(row['xmin']),int(row['ymin']),int(row['xmax']),int(row['ymax'])], [w, h])
                    detections.append(
                        fo.Detection(
                            label=label,
                            bounding_box=box,
                            confidence=score
                        )
                    )
            sample["predictions"] = fo.Detections(detections=detections)
            sample.save()


def filter_view(view):
    """
    条件过滤,具备排序与条件
    """
    return view.sort_by('predictions.detections.confidence', reverse=True)\
        .filter_labels('predictions', F('confidence') > 0.5)


def evaluate(dataset):
    """
    评估模型
    """
    results = dataset.evaluate_detections(
        'predictions',
        gt_field="ground_truth",
        method="open-images",
        eval_key="yolov5",
        compute_mAP=True
    )
    print('\nmAP: %4.f' % results.mAP())


if __name__ == '__main__':
    """
    通过fiftyone实现数据推理可视化，便于针对已标注的数据与模型的推理结论进行对比，从而快速进行模型验证。
    -n (--name) 代表数据集名称，相同数据集名称后续加载将直接通过mongodb读取
    -p (--path)代表数据集的路径，对应的yaml文件件名称为dataset.yaml
    -m (--module)代表需要进行推理的pt文件路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='dataset', help='dataset name')
    parser.add_argument('-p', '--path', type=str, default='../data', help='dataset path')
    parser.add_argument('-m', '--module', type=str, default='yolov5s.pt', help='module path')
    opt = parser.parse_args()
    name = opt.name
    dataset_dir = opt.path
    splits = ["train", "val"]

    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
        for split in splits:
            dataset.add_dir(
                dataset_dir=dataset_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                split=split,
                tags=split,
            )
    else:
        dataset = fo.Dataset(name)
        for split in splits:
            dataset.add_dir(
                dataset_dir=dataset_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                split=split,
                tags=split,
            )

    dataset.tags = ['vms', name]
    dataset.save()
    predictions_view = dataset.take(1000, seed=0)
    predictions(predictions_view, dataset, opt.module)

    bounds = dataset.bounds("predictions.detections.confidence")
    print('置信度范围:')
    print(bounds)

    # evaluate(dataset)

    session = fo.launch_app(filter_view(predictions_view), desktop=True)
    session.wait()