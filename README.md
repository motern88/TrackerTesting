# TrackerTesting
测试各个追踪方案的测试脚本





------

总览：

```python
├──dataset                              # 数据集
├──model								# 模型权重
└──tracker                              # 实现各种具体Tracker
test_tracking.py                        # 使用Tracker推理Dataset的主测试脚本


```



- `test_tracking.py` 是测试的主要脚本

  其中实现`TestTracking()`类，用于加载指定Tracker，并推理指定数据集下所有样本。

  该类的主要外部方法为：`TestTracking.start_test_tracking()`
