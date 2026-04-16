# -*- coding: utf-8 -*-
# @Time    : 2026/03/09 23:23
# @Author  : Yan Zhang
# @FileName: find_file.py
# @Email   : yanzhang1991@cqupt.edu.cn


import os


def find_files(root_path, target='_arch.py'):
    """
    返回指定路径下所有以 _arch.py 结尾的文件的相对路径

    Args:
        root_path: 要搜索的根目录路径

    Returns:
        list: 所有匹配文件的相对路径列表
    """
    arch_files = []

    # 遍历目录树
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            # 检查文件名是否以 _arch.py 结尾
            if filename.endswith(target):
                # 构建完整路径
                full_path = os.path.join(dirpath, filename)
                # 计算相对路径
                relative_path = os.path.relpath(full_path, root_path)
                arch_files.append(relative_path)

    return arch_files

if __name__ == "__main__":
    # 在当前目录下查找
    test_path = "/home/ybb/Project/BasicDFIR/basicsr/archs/"
    result = find_files(test_path, target='_arch.py')
    print(result)
    print(f"找到 {len(result)} 个 _arch.py 文件:")
    for f in result:
        print(f"  - {f}")
