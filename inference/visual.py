import fiftyone as fo
from fiftyone import ViewField as F
import torch
import cv2
import numpy as np


def xyxytoxywh(bbox, size):
    x = bbox[0] / size[0]
    y = bbox[1] / size[1]
    w = (bbox[2] - bbox[0]) / size[0]
    h = (bbox[3] - bbox[1]) / size[1]
    return (x, y, w, h)


def predictions(view, dataset, model_path):
    """
    进行推理，实现将yolo格式标注数据与yolo推理结论进行可视化
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes = dataset.default_classes
    model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=model_path)
    model.to(device)
    model.eval()

    with fo.ProgressBar() as pb:
        for sample in pb(view):
            detections = []
            raw_data = np.fromfile(sample.filepath, dtype=np.uint8)
            image = cv2.imdecode(raw_data, -1)
            h, w, c = image.shape
            results = model(image[..., ::-1])
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
    return view.sort_by(F("predictions.detections").length(), reverse=True)\
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
    name = 'vest_dataset'
    dataset_dir = '../data/vest'
    splits = ["train", "val"]

    if fo.dataset_exists(name):
        dataset = fo.load_dataset(name)
    else:
        dataset = fo.Dataset(name)
        for split in splits:
            dataset.add_dir(
                dataset_dir=dataset_dir,
                dataset_type=fo.types.YOLOv5Dataset,
                split=split,
                tags=split,
            )

    dataset.tags = ['vms', 'vest']
    dataset.save()
    predictions_view = dataset.take(100, seed=2)
    predictions(predictions_view, dataset, '../models/vest.pt')

    evaluate(dataset)

    session = fo.launch_app(filter_view(predictions_view), desktop=True)
    session.wait()