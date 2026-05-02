#!/usr/bin/env python3
"""
PDF上传脚本 - 基于PDF_GALAXAI的后端API

使用前请确保安装依赖:
    pip install requests

使用方法:
    python upload_pdf.py <pdf_file_path> [--ocr-mode auto|force|off] [--host localhost] [--port 8000]

参数:
    pdf_file_path: 要上传的PDF文件路径
    --ocr-mode: OCR模式，可选值: auto, force, off (默认: auto)
    --host: 后端主机 (默认: localhost)
    --port: 后端端口 (默认: 8000)

示例:
    python upload_pdf.py my_paper.pdf
    python upload_pdf.py paper.pdf --ocr-mode force --host 192.168.1.100 --port 8080
"""

import argparse
import os
import sys

try:
    import requests
except ImportError:
    print("错误: 缺少依赖库 'requests'")
    print("请运行: pip install requests")
    sys.exit(1)


def upload_pdf(file_path: str, ocr_mode: str = "auto", host: str = "localhost", port: int = 8000) -> None:
    """
    上传PDF文件到PDF_GALAXAI后端

    Args:
        file_path: PDF文件路径
        ocr_mode: OCR模式 ('auto', 'force', 'off')
        host: 后端主机
        port: 后端端口
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        sys.exit(1)

    # 检查是否为PDF文件
    if not file_path.lower().endswith('.pdf'):
        print(f"错误: 只支持PDF文件，文件 '{file_path}' 不是PDF格式")
        sys.exit(1)

    # 构建API URL
    url = f"http://{host}:{port}/api/upload"

    try:
        # 准备文件数据
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/pdf')}
            data = {'ocr_mode': ocr_mode}

            print(f"正在上传文件: {file_path}")
            print(f"目标URL: {url}")
            print(f"OCR模式: {ocr_mode}")

            # 发送POST请求
            response = requests.post(url, files=files, data=data, timeout=300)  # 5分钟超时

            # 检查响应
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    pdf_id = result.get('pdf_id')
                    print(f"上传成功! PDF ID: {pdf_id}")
                else:
                    print("上传失败: 服务器返回失败状态")
            else:
                print(f"上传失败: HTTP {response.status_code}")
                try:
                    error_detail = response.json().get('detail', response.text)
                    print(f"错误详情: {error_detail}")
                except:
                    print(f"响应内容: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"网络错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"未知错误: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="上传PDF文件到PDF_GALAXAI后端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'file_path',
        help='要上传的PDF文件路径'
    )

    parser.add_argument(
        '--ocr-mode',
        choices=['auto', 'force', 'off'],
        default='auto',
        help='OCR模式 (默认: auto)'
    )

    parser.add_argument(
        '--host',
        default='localhost',
        help='后端主机地址 (默认: localhost)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='后端端口号 (默认: 8000)'
    )

    args = parser.parse_args()

    upload_pdf(
        file_path=args.file_path,
        ocr_mode=args.ocr_mode,
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":
    main()