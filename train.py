import os
import torch
from ultralytics import YOLO

# 设置显存优化环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 更好的错误诊断


def main():
    # 清空显存缓存
    torch.cuda.empty_cache()

    # 加载模型
    model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')

    # 训练配置 - 显存优化参数
    results = model.train(
        data='data.yaml',
        epochs=300,
        patience=100,
        batch=16,  
        imgsz=512,  
        save=True,
        save_period=-1,
        device=0,
        workers=4,  # 减少数据加载工作线程
        project=None,
        name='train_fixed',
        exist_ok=False,
        pretrained='yolov12n.pt', #
        optimizer='SGD',
        verbose=True,
        seed=0,
        deterministic=True,
        single_cls=False,
        val=True,
        amp=True,
        fraction=1.0,
        lr0=0.001,  # 降低学习率
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.95,
        warmup_bias_lr=0.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,  # 减少马赛克增强概率
        mixup=0.0,
        copy_paste=0.0,  # 禁用复制粘贴增强
        auto_augment='randaugment',
        erasing=0.2,  # 减少擦除增强概率
        dropout=0.0,
        gradient_clip_val=1.0,
    )


if __name__ == '__main__':
    # 安装缺失依赖
    try:
        import seaborn
    except ImportError:
        print("安装缺失的seaborn包...")
        import subprocess

        subprocess.run(["pip", "install", "seaborn"], check=True)

    # 运行主函数

    main()
