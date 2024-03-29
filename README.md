# yolotool
汇总了实际项目中所使用的各类工具，便于进行进行项目的快速的验证开展等。

## 基本信息  

### 使用准备
使用前我们需要安装所需的各类依赖从而能够正常使用该工具。  

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### 项目构成
* dvc: 包含数据转换以及数据版本控制脚本，通过deeplake实现数据的版本控制;  
* inference: 包含基于fiftyone的图片推理工具脚本的使用，以及通过视频进行一阶段/两阶段的验证工具脚本;

## 数据版本控制  

### 数据上报

#### 准备工作  
由于通过deeplake实现数据上报前需要针对所需上报的数据进行仓库的创建，否则对应数据的结构与存储方式无法确认，为此
我们需要通过`create_dataset.py`脚本先进行数据仓库的创建，当前框架支持各类符合S3标准接口的存储，这里我们以开源
的minio为例进行介绍，首先我们通过脚本进行数据仓库的创建。  

```bash
python create_dataset.py -b vms -ak 213213 -sk 444a223 -u http://192.168.0.10:9000 -c hat head
```

* -b (--bucket): 数据仓库名称，需要保障每个仓库的名称唯一;  
* -ak (--access_key): S3接口访问令牌;  
* -sk (--secret_key): S3接口访问秘钥;  
* -u (--url): S3访问地址;  
* -c (--classes): 上报数据的分类名称，可支持多个;

完成以上的数据仓库的创建后，就可以在进行标注数据转换后进行上传。  

#### 数据转换上传  
根据实际数据标注的需求，当前主要采用[精灵标注助手](http://www.jinglingbiaozhu.com/)进行数据标注，其默认导出的
标注数据为`pascal-voc`格式，需要将其进行转换从而便于进行模型训练与数据上报。

```bash
python convert_data.py -s ../convertlabels -o ../labels -img ../images -b vms -ak 213213 -sk 444a223 -u http://192.168.0.10:9000 -c hat head
```
* -s (--source): 需要进行转换的数据源文件夹路径
* -o (--output): 转换后标注数据的存储文件夹路径 
* -img (--image_path): 需上传的图片文件夹路径 
* -b (--bucket): 数据仓库名称，需要保障每个仓库的名称唯一 
* -ak (--access_key): S3接口访问令牌 
* -sk (--secret_key): S3接口访问秘钥 
* -u (--url): S3访问地址 
* -c (--classes): 上报数据的分类名称，可支持多个

完成以上操作后即可实现

## 模型验证

### 视频抽帧
针对视频的标注，往往需要针对视频进行抽帧，形成若干的图片后并进行清洗，为了便于进行抽帧，本工具提供了对应的
脚本文件便于进行抽帧，具体文件在`dvc/video_frame_crop.py`中，具体的使用方式如下所示：
```shell
python video_frame_crop.py -s G:/video -o G:/video/images -i 6
```

其中各个参数的说明如下：
* video_name: 输入视频路径 
* num_s: 保存图片的帧率间隔 
* save_path: 抽出的图片保存的位置

### 图片推理
为了便于基于已标注数据图片更好的进行模型的测试论证，采用了`fiftyone`框架提供了可视化能力，为此我们需要
通过`inference/visual.py`进行查看，并根据需要调整其中对应的数据源内容。其由于依赖对应的数据库，故建议
采用专用mongodb存储进行存储，当然也可以直接采用默认的本地文件的形式。  

#### 准备工作

* 1.基于独立mongodb存储

首先我们这里通过采用Docker实现容器化mongodb存储能力，具体脚本如下。
```bash
docker run -d -p 27017:27017 --name mongo -e MONGO_INITDB_ROOT_USERNAME=mongoadmin -e MONGO_INITDB_ROOT_PASSWORD=123456 mongo:4.4
```

完成以上配置后Ubuntu通过`~/.fiftyone/config.json`配置，而对应的Windows系统则通过`C:\Users\[user]\.fiftyone\config.json`
进行配置即可，根据上述启动的mongodb的配置，对应的连接字符串配置如下。  

```json
{
    "database_uri": "mongodb://mongoadmin:123456@127.0.0.1:27017/?authSource=admin"
}
```

关于配置的更多设置可参考[官方文档](https://voxel51.com/docs/fiftyone/user_guide/config.html#)

* 2.准备数据与模型  

框架可以直接支持yolo的标注数据并进行可视化，为此我们可以将对应的数据准备好，放置在对应的文件夹下或者复制到
本项目的目录下均可，并且需要针对其中的YAML文件中数据路径以及文件名进行调整，比如下方的数据存放在项目根目录
的data文件夹下，对应的YAML文件需要改成dataset.yaml并放置在/data/phone文件夹下，其中对应的内容如下。  

```yaml
train: ./images/train
val: ./images/val

nc: 2
names: ['call', 'play']
```

由于框架不仅仅可以查看对应的标注数据，还可以结合模型进行数据推理，从而能够直观的评估模型的实际识别效果，从而
能够针对素材进行调整，以优化模型的实际识别效果。

#### 脚本使用
当前脚本支持通过命令行进行执行，并实现数据的加载读取以及可视化，对应的执行脚本以及对应参数如下。  

```bash
python ./inference/visual.py -n phone-ds -p ../data/phone -m ../models/phone.pt
```

* -n (--name): 代表数据集名称，相同数据集名称后续加载将直接通过mongodb读取  
* -p (--path): 代表数据集的路径，对应的yaml文件件名称为dataset.yaml
* -m (--module): 代表需要进行推理的pt文件路径

#### 模型推理

### 视频推理

#### 单阶段推理
顾名思义，单阶段就是通过单个网络模型完成很对视频、图片的目标检测识别，用户需要选择`oneStage.py`文件，通过修改以下参数完成对应的
识别需求。
* `videoCapture = cv2.VideoCapture('[视频文件路径]')`
* `primary_model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path='[pt模型文件路径]')`
* `if row['name'] == '[需要进行判断识别的异常分类名称]':`

如果需要将识别结论输出到视频文件中，而不仅仅只是实时的观看则可以通过将以下对行的数据去除即可。  
* `18行:out = cv2.VideoWriter('./inference/phone-1.mp4', fourcc, fps, size)`
* `43行:out.write(frame)`
* `53行:out.release()`

#### 两阶段推理
两阶段则是在单阶段的基础上，通过第一阶段识别的结果截取的图片输入给第二阶段的模型从而实现目标检测，如基于人体识别后进行识别人员穿戴
这类场景，针对两阶段的识别需要通过`twoStage.py`文件进行目标检测识别。对应需要修改的内容基本跟单阶段类似，唯一的区别就是需要针对
二阶段的识别结果的分类名称也需要进行配置。  
* `if crop_row['name'] in [分类名序列]`

## 计划
1. 制作基于deeplake兼容[fiftyone](https://voxel51.com/docs/fiftyone/recipes/custom_importer.html)的数据导入源  
2. 