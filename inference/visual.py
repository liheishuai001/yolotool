import fiftyone as fo
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


if __name__ == '__main__':
    fo.config.database_dir = './fiftyonedata'
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
            predictions_view = dataset.take(100, seed=51)
            predictions(predictions_view, dataset, '../models/vest.pt')
            session = fo.launch_app(predictions_view, desktop=True)
            session.show()