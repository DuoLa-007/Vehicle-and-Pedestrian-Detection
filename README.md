1.项目目录

运行前请将文件整理为以下结构：

```text
human\_vehicle\_preprocessing/
├── raw\_data/
│   ├── train\_images/
│   │   ├── 00002.jpg
│   │   └── ...
│   └── train.json
├── processed\_data/              # 运行后自动创建
│   ├── train\_coco.json
│   ├── val\_coco.json
│   └── dataset\_statistics.csv
├── preprocess.py
├── requirements.txt
├── README.md
└── .gitignore
```
2.环境要求与安装

* Windows 10/11 或其他支持 Python 的系统
* Python 3.9 及以上版本（本次运行环境为 Python 3.12.7）
* Pillow（用于读取和检查图片）

在项目根目录打开 PyCharm Terminal、PowerShell 或 CMD，执行：

```bash
python -m pip install -r requirements.txt
```

3. 运行方法

确保当前终端所在目录是包含 `preprocess.py` 和 `requirements.txt` 的项目根目录，然后执行：

```bash
python preprocess.py
```

程序会自动创建 `processed\_data/`，并生成：

|输出文件|内容|
|-|-|
|`train\_coco.json`|训练集 COCO 格式图片与目标标注|
|`val\_coco.json`|验证集 COCO 格式图片与目标标注|
|`dataset\_statistics.csv`|22 个类别的训练/验证目标数量及图片数量统计|

COCO JSON 的顶层包含 `info`、`licenses`、`images`、`annotations` 和 `categories`。`annotations` 中的 `bbox` 使用 `\[x, y, width, height]` 格式，并保留 `truncated`、`occluded` 字段。`images\[].file\_name` 形如 `train\_images/00002.jpg`；读取图片时应将项目中的 `raw\_data/` 作为图片根目录。

4.四项数据预处理

4.1 数据清洗

* 检查标注记录是否为字典，以及 `filename`、`label`、`box` 等必要字段是否存在。
* 检查图片能否正常读取，并获取实际宽高。
* 检查 bbox 坐标是否为有限数值、是否越出图片边界，以及 `xmax > xmin`、`ymax > ymin` 是否成立。
* 根据配置决定是否过滤 Ignore、截断和遮挡目标。
* 对完全重复的标注记录去重。
* 清洗后没有有效目标的图片仍保留在图片样本中，统计时单独记录。

4.2 数据集成

* 从 `train.json` 读取标注列表。
* 按 `filename` 将同一张图片的多个目标标注聚合到一起。
* 将聚合后的标注与 `train\_images/` 中的实际图片文件匹配。
* 建立类别名称到数字 ID 的映射。

4.3 数据变换

* 将原始 `\[xmin, ymin, xmax, ymax]` 转换为 COCO `\[x, y, width, height]`。
* 将类别名称转换为 `category\_id`。
* 将 `truncated`、`occluded` 标记规范为 0/1，并保留在 COCO annotation 中。
* 生成 COCO 标准结构的 `images`、`annotations` 和 `categories` 列表。

4.4 数据归约与划分

* 去除重复标注和非法目标框。
* 根据过滤开关移除指定目标。
* 以图片为单位打乱并划分数据集，同一张图片的全部目标只会进入训练集或验证集中的一个集合。
* 使用固定随机种子，默认按 8:2 划分。

5.本次运行结果

本次运行时采用 `TRAIN\_RATIO=0.8`、`RANDOM\_SEED=42`，并设置 `filter\_truncated=False`、`filter\_occluded=False`、`filter\_ignore=True`。终端统计结果如下：

|统计项目|结果|
|-|-:|
|JSON 标注记录|22,596|
|图片文件|2,600|
|成功匹配图片|2,600|
|未匹配图片/标注|0|
|非法 bbox|1|
|COCO 目标标注|22,595|
|类别数|22|
|训练集图片|2,080|
|验证集图片|520|
|训练集目标|18,036|
|验证集目标|4,559|


