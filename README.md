## 1\. 数据集与实际数据格式

本项目根据下载后的训练集文件组织方式运行，输入为：

* `raw\_data/train\_images/`：训练图片目录，JPG 图片为主，原始图片分辨率为 1280×720。
* `raw\_data/train.json`：训练集标注文件。

`train.json` 顶层包含 `annotations` 列表。每条标注记录主要使用以下字段：

|字段|含义|处理方式|
|-|-|-|
|`filename`|图片路径|统一分隔符并提取图片文件名，用于配对|
|`box`|边界框字典|读取 `xmin`、`ymin`、`xmax`、`ymax`|
|`label`|目标类别名称|转换为 `category\_id`|
|`truncated`|截断标记|转换为 0/1，并保留到 COCO 标注|
|`occluded`|遮挡标记|转换为 0/1，并保留到 COCO 标注|
|`ignore` / `Ignore`|忽略标记（可选）|若源数据存在则按开关处理；当前提供的 `train.json` 未包含该字段|

根据本次实际运行结果：原始 JSON 有 **22,596 条标注记录**，涉及 **2,600 张图片**，识别到 **22 个类别**。图片与标注全部成功匹配，发现并剔除 **1 个非法 bbox**，最终转换 **22,595 个目标标注**。

> 数据集的 `test` 部分没有在本项目中处理，因为本次任务是对带有标注的训练集进行清洗和 8:2 划分；不生成比赛提交文件。

## 2\. 项目目录

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

`raw\_data/` 和 `processed\_data/` 已加入 `.gitignore`，默认不会纳入 Git 版本管理。原始数据通常体积较大，不建议直接上传到 GitHub。若课程明确要求提交处理后的结果文件，可按课程要求单独决定是否上传 `processed\_data/`，但不要上传未经授权公开的数据集原件。

## 3\. 环境要求与安装

* Windows 10/11 或其他支持 Python 的系统
* Python 3.9 及以上版本（本次运行环境为 Python 3.12.7）
* Pillow（用于读取和检查图片）

在项目根目录打开 PyCharm Terminal、PowerShell 或 CMD，执行：

```bash
python -m pip install -r requirements.txt
```

## 4\. 运行方法

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

## 5\. 四项数据预处理

### 5.1 数据清洗

* 检查标注记录是否为字典，以及 `filename`、`label`、`box` 等必要字段是否存在。
* 检查图片能否正常读取，并获取实际宽高。
* 检查 bbox 坐标是否为有限数值、是否越出图片边界，以及 `xmax > xmin`、`ymax > ymin` 是否成立。
* 根据配置决定是否过滤 Ignore、截断和遮挡目标。
* 对完全重复的标注记录去重。
* 清洗后没有有效目标的图片仍保留在图片样本中，统计时单独记录。

### 5.2 数据集成

* 从 `train.json` 读取标注列表。
* 按 `filename` 将同一张图片的多个目标标注聚合到一起。
* 将聚合后的标注与 `train\_images/` 中的实际图片文件匹配。
* 建立类别名称到数字 ID 的映射。

### 5.3 数据变换

* 将原始 `\[xmin, ymin, xmax, ymax]` 转换为 COCO `\[x, y, width, height]`。
* 将类别名称转换为 `category\_id`。
* 将 `truncated`、`occluded` 标记规范为 0/1，并保留在 COCO annotation 中。
* 生成 COCO 标准结构的 `images`、`annotations` 和 `categories` 列表。

### 5.4 数据归约与划分

* 去除重复标注和非法目标框。
* 根据过滤开关移除指定目标。
* 以图片为单位打乱并划分数据集，同一张图片的全部目标只会进入训练集或验证集中的一个集合。
* 使用固定随机种子，默认按 8:2 划分。

## 6\. 类别映射

程序根据类别名称的字母排序，从 1 开始分配 `class\_id`。本次实际运行得到的映射如下：

|class\_id|类别名称|class\_id|类别名称|
|-:|-|-:|-|
|1|BiCyclist|12|MotorCyclist|
|2|Bike|13|Motorcycle|
|3|CampusBus|14|OtherCar|
|4|Car|15|Pedestrian|
|5|EngineTruck|16|PersonSitting|
|6|HeavyTruck|17|Pickup|
|7|LargeBus|18|TricycleClosed|
|8|LightTruck|19|TricycleOpenHuman|
|9|MMcar|20|TricycleOpenMotor|
|10|Machineshop|21|Truck|
|11|MediumBus|22|van|

如果课程提供了官方类别 ID 顺序，应以课程要求为准，并相应调整脚本中的类别映射规则。

## 7\. 可修改配置

在 `preprocess.py` 文件顶部可以修改以下参数：

```python
TRAIN\_RATIO = 0.8
RANDOM\_SEED = 42

filter\_truncated = False
filter\_occluded = False
filter\_ignore = True
```

* `TRAIN\_RATIO`：训练集比例，默认 0.8，验证集比例为 0.2。
* `RANDOM\_SEED`：随机种子，默认 42，保证相同输入和配置下划分可复现。
* `filter\_truncated`：设为 `True` 时过滤截断目标。
* `filter\_occluded`：设为 `True` 时过滤遮挡目标。
* `filter\_ignore`：设为 `True` 时过滤源数据中标记为 Ignore 的目标。当前提供的 `train.json` 没有 Ignore 字段，因此本次运行没有因此过滤目标。

## 8\. 本次运行结果

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

以上是本次数据和配置下的运行记录；更换数据或修改过滤参数后，统计数字可能变化。

## 9\. 说明与限制

* 本项目仅处理带标注的训练数据，不训练模型、不使用数据库、不生成比赛提交文件。
* 脚本不会修改原始图片和 `train.json`。
* 类别 ID 按名称排序生成，不应默认等同于平台内部或其他模型配置中的官方类别 ID。
* 过滤开关只影响目标标注；若某张图片过滤后没有有效目标，图片本身仍会保留在划分结果中。
* 本项目输出的 COCO `file\_name` 是相对于 `raw\_data/` 的路径，不是相对于项目根目录的路径。



