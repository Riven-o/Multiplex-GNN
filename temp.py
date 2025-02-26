import os
import filecmp

def compare_directories(path1, path2):
    matches = []
    # 获取规范化路径确保一致性
    path1 = os.path.normpath(path1)
    path2 = os.path.normpath(path2)

    # 递归遍历第一个目录
    for root, dirs, files in os.walk(path1):
        # 获取相对路径
        relative_path = os.path.relpath(root, path1)
        # 构建对应目录的路径
        compare_root = os.path.join(path2, relative_path)

        # 比较文件
        for filename in files:
            file1 = os.path.join(root, filename)
            file2 = os.path.join(compare_root, filename)

            # 检查文件是否存在且为普通文件
            if os.path.isfile(file1) and os.path.isfile(file2):
                # 使用文件内容比较（shallow=False表示比较内容而不仅仅是元数据）
                if filecmp.cmp(file1, file2, shallow=False):
                    # 记录相对路径
                    relative_file = os.path.join(relative_path, filename)
                    matches.append(relative_file)

    return matches

if __name__ == "__main__":
    path1 = "/home/mufan/mohan/gnn/dataset/2/subjects/100206"
    path2 = "/home/mufan/mohan/gnn/dataset/2/subjects/100307"

    # 输入路径检查
    if not (os.path.isdir(path1) and os.path.isdir(path2)):
        print("错误：输入的路径必须都是有效的目录")
        exit(1)

    # 执行比较
    matched_files = compare_directories(path1, path2)

    # 输出结果
    if matched_files:
        print("以下文件内容完全相同：")
        for file_path in matched_files:
            print(f"- {file_path}")
        print(f"共找到 {len(matched_files)} 个相同文件")
    else:
        print("没有找到内容相同的文件")