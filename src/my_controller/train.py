##训练脚本
from ultralytics import YOLO


def train_yolov11_core():
    # ---------------------- 核心参数配置----------------------
    # 1. YOLOv11 网络结构配置文件路径
    model_yaml_path = "ultralytics/cfg/models/11/yolo11.yaml"
    # 2. 预训练权重文件名
    pre_model_name = "yolo11n.pt"
    # 3. 数据集配置文件路径
    data_yaml_path = "ultralytics/cfg/datasets/kon.yaml"

    # ---------------------- 可选训练参数----------------------
    epochs = 100
    imgsz = 640
    output_dir = "result"
    batch_size = -1
    workers = 1

    try:
        model = YOLO(model_yaml_path)

        model = model.load(pre_model_name)  # 也可简写为：model = YOLO(model_yaml_path).load(pre_model_name)

        # 定义任务名称（单独提取，方便后续打印，避免硬编码不一致）
        task_name = "kon_yolov11_core_train"
        train_results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=output_dir,
            name=task_name,  # 使用定义好的任务名称
            cache="ram",  # 缓存数据集到内存，提升训练速度（适合小数据集）
            workers=workers,
            exist_ok=True,
            save=True,  # 保存最佳/最后模型
            verbose=True
        )

        # 训练完成提示（修复：使用实际的 task_name，而非固定字符串）
        print(f"\n? 训练全部完成！")
        print(f"? 训练结果保存路径：{output_dir}/{task_name}")
        print(f"? 最佳模型路径：{output_dir}/{task_name}/weights/best.pt")

    # 补充异常处理（避免程序意外崩溃，方便排查问题）
    except FileNotFoundError as e:
        print(f"\n? 错误：找不到指定文件 - {e}")
        print("请检查 model_yaml_path、data_yaml_path 路径是否正确，或预训练权重是否下载成功")
    except Exception as e:
        print(f"\n? 训练过程中出现未知错误 - {e}")


if __name__ == "__main__":
    # 运行核心训练函数
    train_yolov11_core()