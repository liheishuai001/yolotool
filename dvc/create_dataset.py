import argparse
import deeplake


if __name__ == '__main__':
    """
    创建数据仓储，需要在进行上传前首先进行创建才可上传数据。
    -b (--bucket): 数据仓库名称，需要保障每个仓库的名称唯一
    -ak (--access_key): S3接口访问令牌
    -sk (--secret_key): S3接口访问秘钥
    -u (--url): S3访问地址
    -c (--classes): 上报数据的分类名称，可支持多个
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bucket', type=str, default='yolo', help='s3 bucket name')
    parser.add_argument('-ak', '--access_key', type=str, required=True, help='s3 access key')
    parser.add_argument('-sk', '--secret_key', type=str, required=True, help='s3 access secret key')
    parser.add_argument('-u', '--url', type=str, default='', help='s3 endpoint url')
    parser.add_argument('-c', '--classes', type=str, nargs='+', help='model class name')
    opt = parser.parse_args()
    if not opt.url.startswith('http'):
        raise ValueError(f'the url is invalid {opt.url}')
    ds = deeplake.empty(f's3://{opt.bucket}', creds= {
        'aws_access_key_id': opt.access_key,
        'aws_secret_access_key': opt.secret_key,
        'endpoint_url': opt.url
    })
    with ds:
        ds.create_tensor('images', htype='image', sample_compression='jpeg')
        ds.create_tensor('labels', htype='class_label', class_names= opt.classes)
        ds.create_tensor('boxes', htype='bbox')
        ds.boxes.info.update(coords={'type': 'fractional', 'mode': 'LTWH'})