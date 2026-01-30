##推理脚本：支持图片/视频混合输入，自动识别类型
from ultralytics import YOLO
import torch
import os

# 支持的图片/视频格式（可根据需要扩展）
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')


def get_yolov11_inference_config(device_type: str = None) -> dict:
    """
    根据设备类型自动获取YOLOv11最优推理配置（batch + half）
    :param device_type: 指定设备"gpu"/"cpu"，为None时自动检测
    :return: 包含yolov11推理所需参数的配置字典
    """
    # 步骤1：自动检测设备（优先GPU）
    if device_type is None:
        has_gpu = torch.cuda.is_available()
        device_type = "gpu" if has_gpu else "cpu"

    # 步骤2：根据设备配置核心参数（严格遵循需求）
    device_type = device_type.lower()
    inference_config = {}

    if device_type == "gpu":
        # GPU配置：batch=1（避免显存溢出），half=True（开启FP16，显存减半、提速）
        inference_config["batch"] = 1
        inference_config["half"] = True
        inference_config["device"] = 0  # 指定使用第0块GPU（YOLOv11识别）
        print(f"已配置GPU推理参数：batch=1, half=True, device=0")

    elif device_type == "cpu":
        # CPU配置：batch=2-4（默认2，可调整），half=False（不支持FP16）
        inference_config["batch"] = 2
        inference_config["half"] = False
        inference_config["device"] = "cpu"  # 指定使用CPU（YOLOv11识别）
        print(f"已配置CPU推理参数：batch=2, half=False, device=cpu")

    else:
        raise ValueError("不支持的设备类型，仅支持 'gpu' 或 'cpu'")

    # 步骤3：添加CPU batch调整方法（限制2-4，满足小幅提速需求）
    def adjust_cpu_batch(new_batch: int):
        if device_type != "cpu":
            raise PermissionError("仅CPU设备可调整batch大小")
        if 2 <= new_batch <= 4:
            inference_config["batch"] = new_batch
            print(f"CPU batch已调整为：{new_batch}")
        else:
            raise ValueError("CPU batch必须在2-4之间（闭区间）")

    inference_config["adjust_cpu_batch"] = adjust_cpu_batch
    return inference_config


def _judge_input_type(source_path: str) -> str:
    """
    内部辅助函数：判断单个输入文件的类型（图片/视频）
    :param source_path: 输入文件路径
    :return: "image" / "video"
    """
    file_ext = os.path.splitext(source_path)[1].lower()
    if file_ext in SUPPORTED_IMAGE_FORMATS:
        return "image"
    elif file_ext in SUPPORTED_VIDEO_FORMATS:
        return "video"
    else:
        raise ValueError(f"不支持的文件格式：{file_ext}，支持的图片格式{SUPPORTED_IMAGE_FORMATS}，支持的视频格式{SUPPORTED_VIDEO_FORMATS}")


def yolov11_multi_inference(model_path: str, source_paths: list, config: dict):
    """
    调用YOLOv11进行多类型推理（支持图片/视频，自动识别类型）
    适配低版本Ultralytics，添加stream=True解决视频内存溢出警告
    :param model_path: YOLOv11模型路径（如'yolov11n.pt'、'yolov11s.pt'）
    :param source_paths: 待推理的文件路径列表（可混合图片/视频，推荐按类型分组）
    :param config: 由get_yolov11_inference_config返回的配置字典
    """
    # 步骤1：加载YOLOv11模型（指定设备）
    try:
        model = YOLO(model_path)
        print(f"\n成功加载YOLOv11模型：{model_path}")
    except Exception as e:
        raise Exception(f"模型加载失败：{e}")

    # 步骤2：提取配置参数（对接YOLOv11的predict方法）
    batch_size = config["batch"]
    use_half = config["half"]
    device = config["device"]

    # 步骤3：按文件类型分组处理（优化推理效率，避免混合类型报错）
    image_sources = []
    video_sources = []
    for path in source_paths:
        if not os.path.exists(path):
            print(f"警告：文件不存在，跳过 -> {path}")
            continue
        file_type = _judge_input_type(path)
        if file_type == "image":
            image_sources.append(path)
        else:
            video_sources.append(path)

    all_results = {}  # 存储所有推理结果，按类型分类返回

    # 子步骤3.1：处理图片推理（保留原有批处理逻辑，无需stream=True，批量返回结果）
    if image_sources:
        print(f"\n===== 开始图片推理 =====")
        print(f"共{len(image_sources)}张图片，批处理大小：{batch_size}")
        image_results = model.predict(
            source=image_sources,
            batch=batch_size,
            half=use_half,
            device=device,
            verbose=False,
            save=True,  # 保存推理结果图片（默认保存在runs/detect/predict目录）
            show=False,
            conf=0.25  # 置信度阈值，过滤低置信度目标
            # 图片推理不使用stream=True，保持批量返回，方便对应图片路径解析结果
        )

        # 解析图片推理结果
        for idx, (img_path, result) in enumerate(zip(image_sources, image_results)):
            det_count = len(result.boxes) if result.boxes is not None else 0
            print(f"图片 {idx + 1}（{img_path}）：检测到 {det_count} 个目标")

        all_results["images"] = {
            "sources": image_sources,
            "results": image_results
        }

    # 子步骤3.2：处理视频推理（添加stream=True，流式逐帧处理，解决内存溢出）
    if video_sources:
        print(f"\n===== 开始视频推理 =====")
        print(f"共{len(video_sources)}个视频，流式逐帧推理（避免内存溢出，保存完整视频结果）")
        video_results = []
        for vid_path in video_sources:
            print(f"\n正在处理视频：{vid_path}")
            # 启用stream=True，返回生成器，逐帧加载结果
            vid_result_generator = model.predict(
                source=vid_path,
                batch=1,  # 视频推理强制batch=1，避免帧顺序混乱
                half=use_half,
                device=device,
                verbose=False,
                save=True,  # 低版本中，save=True自动保存完整视频
                show=False,
                conf=0.25,  # 置信度阈值，过滤低置信度目标
                stream=True  # 核心修改：启用流式处理，消除内存溢出警告
            )

            # 遍历生成器，逐帧处理结果（可选：统计总检测帧数以反馈进度）
            frame_count = 0
            vid_frames_results = []
            for frame_result in vid_result_generator:
                frame_count += 1
                vid_frames_results.append(frame_result)
                # 可选：每100帧打印一次进度，避免日志冗余
                if frame_count % 100 == 0:
                    print(f"  已处理 {frame_count} 帧")

            # 存储该视频的完整帧结果和路径
            video_results.append((vid_path, vid_frames_results))
            print(f"视频 {vid_path} 处理完成：共处理 {frame_count} 帧，结果已保存为完整视频文件")

        all_results["videos"] = {
            "sources": video_sources,
            "results": video_results
        }

    # 步骤4：返回汇总结果
    if all_results:
        print(f"\n===== 所有推理任务完成 =====")
        print(f"处理完成：图片{len(image_sources)}张，视频{len(video_sources)}个")
        print(f"推理结果默认保存路径：./runs/detect/predict/")
    else:
        print(f"\n===== 无有效文件进行推理 =====")

    return all_results


# ---------------------- 测试使用（直接运行即可） ----------------------
if __name__ == "__main__":
    # 1. 配置参数（自动检测设备）
    yolov11_config = get_yolov11_inference_config()

    # 2. （可选）CPU环境下调整batch为4（最大化CPU提速效果）
    try:
        yolov11_config["adjust_cpu_batch"](4)
    except Exception as e:
        print(f"无需调整CPU batch：{e}")

    # 3. 准备待推理文件（支持图片+视频混合，替换为你的文件路径）
    test_source_paths = [
        # 图片文件（保留你原有测试图片）
        r"E:\neutral\yolov11\ultralytics-8.3.163\datasets\kon_dataset\images\test\0ea4b4d7-b73f-4e3e-a89b-10cf69d80196.png",
        r"E:\neutral\yolov11\ultralytics-8.3.163\datasets\kon_dataset\images\test\07699765-c6d1-4c04-bf30-2eaa61b303dc.png",
        r"E:\neutral\d043ad4bd11373f0cc81c6d5a00f4bfbfbed0439.jpg",
        # 视频文件（添加你的视频路径，示例格式）
         r"E:\neutral\kon_test_dem0.mp4",
        # r"E:\neutral\yolov11\demo.mkv",
    ]

    # 4. 调用YOLOv11多类型推理（适配低版本，解决视频内存溢出警告）
    try:
        inference_results = yolov11_multi_inference(
            model_path=r"E:\neutral\yolov11\ultralytics-8.3.163\result\kon_yolov11_core_train\weights\best.pt",
            source_paths=test_source_paths,
            config=yolov11_config,
        )
    except Exception as e:
        print(f"\n推理失败：{e}")