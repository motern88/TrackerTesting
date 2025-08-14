## TrackerTesting

测试各个追踪方案的测试脚本



### 1. Overview

总览：

```python
├──dataset/                             # 数据集
├──model/								# 模型权重
├──output/                              # 可视化等输出结果
└──tracker/                             # 实现各种具体Tracker
test_tracking.py                        # 使用Tracker推理Dataset的主测试脚本


```



- `test_tracking.py` 是测试的主要脚本

  其中实现`TestTracking()`类，用于加载指定Tracker，并推理指定数据集下所有样本。

  该类的主要外部方法为：`TestTracking.start_test_tracking()`





### 2. Quick Start

1.配置 `test_tracking.py` 脚本中执行时的 `TestTracking()` 类初始化（`if __name__ == "__main__":`下）：

向 `TestTracking()` 初始化时传入参数：

- `test_dataset_path` 测试数据集目录

- `tracker` 要测试的追踪器

  在 `tracker/` 目录下实现一系列可以测试的追踪器

- `output_dir` 可视化结果保存的路径



2.在TrackerTesting根目录下运行

```
python test_tracking.py
```

