import argparse
import deeplake

'''
创建数据仓储，需要在进行上传前首先进行创建才可上传数据。
bucket：所创建的桶名称，建议按照项目名与模型名对应，如vms/vest这种形式

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bucket', type=str, default='yolo', help='s3 bucket name')
    parser.add_argument('-ak', '--access_key', type=str, required=True, help='s3 access key')
    parser.add_argument('-sk', '--secret_key', type=str, required=True, help='s3 access secret key')
    parser.add_argument('-u', '--url', type=str, default='', help='s3 endpoint url')
    parser.add_argument('-c', '--classes', type=str, nargs='+', help='pt class name')
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