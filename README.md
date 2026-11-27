# Vehicle-and-Pedestrian-Detection

DataFountain 552 人车目标检测竞赛项目。项目目标是将赛题标注转换为 YOLO 格式，完成训练/验证数据划分，并将模型预测转换为竞赛提交 JSON。

> **当前状态：数据准备代码与项目文档已整理；模型训练、验证指标、测试集推理和正式竞赛提交尚未在本仓库中验证。请勿将本仓库视为已取得竞赛成绩的完整训练成果。**

## 目录

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── prepare_dataset.py
│   └── make_submission.py
├── docs/
│   ├── dataset_and_method.md
│   └── training_checklist.md
├── data/
│   └── README.md
└── weights/
    └── README.md
```

## 1. 赛题数据

请从 [DataFountain 竞赛页面](https://www.datafountain.cn/competitions/552) 按竞赛规则获取数据。将下载文件放在本地项目的 `raw/` 目录：

```text
raw/
├── train_dataset.zip
├── test_dataset.zip
└── submit_example.json
```

**不要将原始图片、标注文件或测试数据上传到公开 GitHub 仓库。** 数据仅按竞赛授权范围使用；请以竞赛页面最新规则为准。

## 2. 环境准备

建议使用 Linux 或 WSL2 Ubuntu。OneFlow YOLOv5 上游项目说明了其平台支持情况；请先核对当前 CUDA、驱动与 OneFlow 版本的兼容性。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell 激活命令：

```powershell
.venv\\Scripts\\Activate.ps1
```

## 3. 数据预处理

在项目根目录执行：

```bash
python scripts/prepare_dataset.py --raw_dir raw --out_dir dataset --val_ratio 0.15
```

脚本将解压数据、按图片划分训练/验证集、转换 YOLO 标签、生成类别映射和审计报告。固定随机种子为 42，便于复现。

生成的 `dataset/` 属于本地数据产物，已加入 `.gitignore`，不应上传到公开仓库。

## 4. 模型训练

赛题涉及 OneFlow 框架。可参考 [OneFlow YOLOv5 上游仓库](https://github.com/Oneflow-Inc/one-yolov5) 与 [官方文档](https://start.oneflow.org/oneflow-yolo-doc)。

克隆并检查上游仓库当前版本的安装说明和训练参数后，再确定具体训练命令。不要直接假设不同版本的参数、预训练权重格式或 CUDA 组合完全兼容。建议先用小 batch 做单轮/短程试跑，确认数据加载、显存和损失计算正常。

本仓库暂不包含未经验证的训练命令、权重或指标。

## 5. 生成竞赛提交文件

模型推理后，需先导出 YOLO 检测文本（每行格式：`class xc yc w h confidence`，坐标为归一化值），再执行：

```bash
python scripts/make_submission.py \
  --pred_dir path/to/predictions \
  --image_dir dataset/images/test \
  --classes dataset/classes.json \
  --out submission.json
```

脚本生成顶层包含 `annotations` 的 JSON。请使用少量图片先核对文件名、类别大小写、坐标范围及字段格式，并以竞赛官方提交说明为最终准则。

## 6. 复现与报告

建议记录：
- Python、CUDA、驱动、OneFlow 和上游代码版本；
- 数据划分随机种子及数据审计结果；
- 模型配置、输入尺寸、batch size、epoch、优化器和学习率；
- 验证集指标、训练日志和权重来源；
- 测试集预测及提交文件校验结果。

目前没有在本仓库中发布训练结果。只有完成实际训练和验证后，才应在实验报告中填写指标。

## 许可证与引用

竞赛数据的使用权受竞赛规则约束，不因本仓库公开而获得额外授权。OneFlow YOLOv5 是独立上游项目，请遵守其仓库许可证与引用要求。本仓库的数据处理脚本和文档不代表对上游代码的再分发。
