#该脚本用于数据集的处理与划分
"""
数据集划分工具
功能：将images文件夹中的PNG图片和labels文件夹中的TXT标签文件随机划分为训练集、验证集和测试集
注意：只复制文件，不修改原始文件
"""

import os
import random
import shutil
from tqdm import tqdm  # 用于显示进度条


def split_dataset(images_dir='images',
                  labels_dir='labels',
                  output_dir='test_dataset',
                  train_ratio=0.7,
                  valid_ratio=0.2,
                  test_ratio=0.1):
    """
    随机划分数据集并复制文件到指定目录结构

    参数说明:
    images_dir: 原始图片文件夹路径（存放.png文件）
    labels_dir: 原始标签文件夹路径（存放.txt文件）
    output_dir: 输出文件夹路径
    train_ratio: 训练集比例 (0-1之间的小数)
    valid_ratio: 验证集比例 (0-1之间的小数)
    test_ratio: 测试集比例 (0-1之间的小数)
    """

    print("=" * 60)
    print("开始划分数据集")
    print("=" * 60)

    # 1. 检查比例之和是否为1
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:  # 允许微小的浮点数误差
        print(f"??  注意：比例之和为{total_ratio:.3f}，应该为1.0")

        # 自动调整比例，使其和为1
        scale_factor = 1.0 / total_ratio
        train_ratio *= scale_factor
        valid_ratio *= scale_factor
        test_ratio *= scale_factor

        print(f"? 已自动调整为: 训练集={train_ratio:.3f}, 验证集={valid_ratio:.3f}, 测试集={test_ratio:.3f}")

    # 2. 检查原始文件夹是否存在
    print("\n? 检查文件夹...")
    if not os.path.exists(images_dir):
        print(f"? 错误：图片文件夹 '{images_dir}' 不存在！")
        return False

    if not os.path.exists(labels_dir):
        print(f"? 错误：标签文件夹 '{labels_dir}' 不存在！")
        return False

    print("? 文件夹检查通过")

    # 3. 获取所有PNG图片文件
    print("\n? 扫描图片文件...")
    # 获取所有.png文件（包括.PNG，统一转换为小写判断）
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]

    if not image_files:
        print(f"? 错误：在 '{images_dir}' 中没有找到.png图片文件！")
        return False

    print(f"? 找到 {len(image_files)} 个PNG图片文件")

    # 4. 检查对应的标签文件是否存在
    print("\n? 检查对应的标签文件...")
    valid_files = []  # 存储有效的文件对
    missing_labels = 0  # 统计缺失标签的数量

    # 使用tqdm显示进度条
    for img_file in tqdm(image_files, desc="检查标签文件"):
        # 获取不带扩展名的文件名（例如：'abc.png' -> 'abc'）
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_file)

        if os.path.exists(label_path):
            # 如果标签文件存在，添加到有效文件列表
            valid_files.append({
                'image': img_file,
                'label': label_file,
                'base_name': base_name
            })
        else:
            missing_labels += 1

    # 报告缺失标签的情况
    if missing_labels > 0:
        print(f"??  警告：有 {missing_labels} 个图片没有对应的标签文件，这些文件将被忽略")

    if not valid_files:
        print("? 错误：没有找到任何有效的图片-标签对！")
        return False

    print(f"? 找到 {len(valid_files)} 个有效的图片-标签对")

    # 5. 随机打乱数据
    print("\n? 随机打乱数据顺序...")
    random.shuffle(valid_files)  # 随机打乱顺序，确保划分的随机性

    # 6. 计算各集合的大小
    total_files = len(valid_files)
    train_count = int(total_files * train_ratio)  # 训练集数量
    valid_count = int(total_files * valid_ratio)  # 验证集数量
    test_count = total_files - train_count - valid_count  # 测试集数量

    # 确保测试集数量不为负数
    if test_count < 0:
        print("? 错误：计算出的测试集数量为负数，请检查比例设置！")
        return False

    print(f"\n? 划分结果:")
    print(f"   训练集: {train_count} 个文件 ({train_ratio * 100:.1f}%)")
    print(f"   验证集: {valid_count} 个文件 ({valid_ratio * 100:.1f}%)")
    print(f"   测试集: {test_count} 个文件 ({test_ratio * 100:.1f}%)")

    # 7. 划分数据
    train_files = valid_files[:train_count]  # 前train_count个作为训练集
    valid_files_list = valid_files[train_count:train_count + valid_count]  # 中间部分作为验证集
    test_files = valid_files[train_count + valid_count:]  # 剩余部分作为测试集

    # 8. 创建输出目录结构
    print("\n? 创建目录结构...")
    dirs_to_create = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'valid'),
        os.path.join(output_dir, 'images', 'test'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'valid'),
        os.path.join(output_dir, 'labels', 'test')
    ]

    # 创建所有需要的目录
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)  # exist_ok=True表示如果目录已存在也不会报错
        print(f"   创建目录: {dir_path}")

    # 9. 定义复制文件的函数
    def copy_files(file_list, subset_name, file_type):
        """
        复制文件到对应的子集文件夹

        参数:
        file_list: 文件列表
        subset_name: 子集名称 ('train', 'valid', 'test')
        file_type: 文件类型 ('images' 或 'labels')
        """
        if not file_list:
            print(f"??  警告：{subset_name}集没有{file_type}文件，跳过复制")
            return

        # 根据文件类型选择源文件夹和目标文件夹
        if file_type == 'images':
            src_dir = images_dir
            dst_dir = os.path.join(output_dir, 'images', subset_name)
        else:  # labels
            src_dir = labels_dir
            dst_dir = os.path.join(output_dir, 'labels', subset_name)

        # 使用tqdm显示复制进度
        for item in tqdm(file_list, desc=f"复制{file_type}-{subset_name}集"):
            # 获取文件名
            if file_type == 'images':
                filename = item['image']
            else:
                filename = item['label']

            # 构建源文件和目标文件的完整路径
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)

            # 复制文件（shutil.copy2会保留文件元数据，如创建时间）
            shutil.copy2(src_path, dst_path)

    # 10. 复制各集合的文件
    print("\n? 开始复制文件...")

    # 复制训练集文件
    print("\n? 复制训练集文件...")
    copy_files(train_files, 'train', 'images')
    copy_files(train_files, 'train', 'labels')

    # 复制验证集文件
    print("\n? 复制验证集文件...")
    copy_files(valid_files_list, 'valid', 'images')
    copy_files(valid_files_list, 'valid', 'labels')

    # 复制测试集文件
    print("\n? 复制测试集文件...")
    copy_files(test_files, 'test', 'images')
    copy_files(test_files, 'test', 'labels')

    # 11. 显示完成信息
    print("\n" + "=" * 60)
    print("? 数据集划分完成！")
    print("=" * 60)

    print(f"\n? 结果已保存到: {output_dir}")
    print("\n? 目录结构:")
    print(f"  {output_dir}/")
    print(f"    ├── images/")
    print(f"    │   ├── train/  (包含 {len(train_files)} 个PNG图片文件)")
    print(f"    │   ├── valid/  (包含 {len(valid_files_list)} 个PNG图片文件)")
    print(f"    │   └── test/   (包含 {len(test_files)} 个PNG图片文件)")
    print(f"    └── labels/")
    print(f"        ├── train/  (包含 {len(train_files)} 个TXT标签文件)")
    print(f"        ├── valid/  (包含 {len(valid_files_list)} 个TXT标签文件)")
    print(f"        └── test/   (包含 {len(test_files)} 个TXT标签文件)")

    # 12. 保存划分信息到文件
    print("\n? 保存划分信息...")
    info_file = os.path.join(output_dir, 'split_info.txt')
    try:
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("数据集划分信息\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"原始图片文件夹: {images_dir}\n")
            f.write(f"原始标签文件夹: {labels_dir}\n")
            f.write(f"输出文件夹: {output_dir}\n\n")
            f.write(f"总有效文件数: {total_files}\n")
            f.write(f"训练集比例: {train_ratio:.3f} ({train_count}个文件)\n")
            f.write(f"验证集比例: {valid_ratio:.3f} ({len(valid_files_list)}个文件)\n")
            f.write(f"测试集比例: {test_ratio:.3f} ({len(test_files)}个文件)\n\n")

            f.write("训练集文件列表:\n")
            f.write("-" * 30 + "\n")
            for item in train_files:
                f.write(f"  {item['base_name']}\n")

            f.write("\n验证集文件列表:\n")
            f.write("-" * 30 + "\n")
            for item in valid_files_list:
                f.write(f"  {item['base_name']}\n")

            f.write("\n测试集文件列表:\n")
            f.write("-" * 30 + "\n")
            for item in test_files:
                f.write(f"  {item['base_name']}\n")

        print(f"? 划分详细信息已保存到: {info_file}")
    except Exception as e:
        print(f"??  警告：无法保存划分信息文件: {e}")

    print("\n? 所有操作已完成！")
    print("=" * 60)

    return True


# 主程序入口
if __name__ == "__main__":
    # ==================== 在这里修改参数 ====================
    # 原始文件夹路径（默认是当前目录下的images和labels文件夹）
    IMAGES_DIR = "images"  # 存放PNG图片的文件夹路径
    LABELS_DIR = "labels"  # 存放TXT标签的文件夹路径

    # 输出文件夹
    OUTPUT_DIR = "test_dataset"  # 结果保存的文件夹

    # 划分比例（三个比例加起来应该等于1）
    TRAIN_RATIO = 0.7  # 训练集比例 (70%)
    VALID_RATIO = 0.2  # 验证集比例 (20%)
    TEST_RATIO = 0.1  # 测试集比例 (10%)
    # ======================================================

    # 显示参数信息
    print("? 参数设置:")
    print(f"  图片文件夹: {IMAGES_DIR}")
    print(f"  标签文件夹: {LABELS_DIR}")
    print(f"  输出文件夹: {OUTPUT_DIR}")
    print(f"  划分比例: 训练集={TRAIN_RATIO}, 验证集={VALID_RATIO}, 测试集={TEST_RATIO}")

    # 运行划分函数
    success = split_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        test_ratio=TEST_RATIO
    )

    if success:
        print("\n? 脚本执行成功！")
    else:
        print("\n? 脚本执行失败，请检查错误信息。")