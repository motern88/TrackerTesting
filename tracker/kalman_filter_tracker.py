'''
这里实现kalman滤波追踪器
使用基于黑白摄像头模型的球的YOLO目标检测（不区分球类别），再使用kalman滤波追踪来将第一帧的球类别追踪到后续的每一帧上。

具体地，YOLO 检测器定位球体中心坐标，卡尔曼滤波基于匀速运动模型预测目标轨迹并校正观测噪声，
匈牙利算法（配合欧氏距离阈值）完成检测框与预测轨迹的最优匹配。
针对新出现目标、短暂遮挡和轨迹丢失等情况，采用 ID 自增分配、预测维持和连续丢帧剔除等策略，确保跟踪结果的连续性和鲁棒性。

关键参数（如噪声权重、匹配阈值）需要经手动实验调优，平衡检测精度与运动预测的可靠性。

额外环境依赖：
    pip install ultralytics
    pip install filterpy

'''
from tracker.base_tracker import BaseTracker
from ultralytics import YOLO

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter  # 导入卡尔曼滤波器

class KalmanFilterTracker(BaseTracker):
    '''
    Tracker需要实现以下方法：

    - 清除所有历史轨迹
        self.clear_history_trajectory()
    - 获取历史轨迹
        self.get_history_trajectory()
    - 处理（推理）新一帧
        self.process_frame(frame)

    以上部分方法由BaseTracker实现，部分方法由本子类实现。
    本子类主要实现基于YOLO检测+kalman滤波的追踪算法
    '''
    def __init__(self):
        '''
        继承父类初始化的缓存结构：
        self.frames = [] : 用于存放所有原始帧数据的列表
        self.predict_results = {} : 用于存放每一帧预测结果的字典
            {"frame_idx": predict_result_list}  key为帧索引，predict_result_list为预测结果列表
            predict_result_list = [{"id":1, "location": {"x": 160.52359, "y": 119.27372}},...]
        '''
        super().__init__()
        # 加载YOLO模型
        self.model = YOLO("model/train_ir_9_640.pt")

        self.trackers = {}  # 存储每个ID的卡尔曼滤波器
        self.next_id = 1  # 下一个可用的ID


    # 对新帧进行处理
    def process_frame(self, frame):
        '''
        对每一新帧的处理：
        1. 对新帧进行YOLO检测（只检测球，不分类别）
        2. 使用卡尔曼滤波对检测到的球位置进行预测+更新
        3. 存储预测结果到 self.predict_results
        '''
        # 追加原始帧
        self.frames.append(frame)
        frame_idx = len(self.frames) - 1

        # 1. YOLO检测
        detect_results = self.run_yolo_detection(frame)
        # detect_results 结构示例：
        # [{"location": {"x": 121.0, "y": 249.9}}, ...] 需要将检测框转化为中心坐标
        # print(f"[Tracker] 当前帧检测结果：\n{detect_results}")

        # 为第一帧分配ID
        if frame_idx == 0:
            tracked_results = self.init_tracking_ids(detect_results)  # 为第一帧检测结果直接分配ID作为初始追踪结果

        # 为后续帧进行卡尔曼滤波预测+更新
        else:
            # 2. 卡尔曼滤波预测 + 更新
            tracked_results = self.kalman_update(detect_results)
            # tracked_results 结构示例：
            # [{"id": 1, "location": {"x": 121.0, "y": 249.9}}, ...]
            # 这里要求卡尔曼滤波更新的ID等于实际类别ID，后续不做单独的类别映射


        # 3. 缓存预测结果
        self.predict_results[frame_idx] = tracked_results

        return tracked_results

    # 上：Tracker主要方法
    # ------------------------------------------------------------------------------------------------------
    # 下：辅助方法

    # yolo进行检测返回检测结果
    def run_yolo_detection(self, frame):
        '''
        使用yolo模型进行检测，这里是黑白模型，其中cls为2代表球类别
        返回格式：
        [
            {"location": {"x": float, "y": float}},
            {"location": {"x": float, "y": float}},
            ...
        ]
        '''
        detect_results = []  # 初始化空的检测结果列表

        # 进行检测
        results = self.model([frame], stream=True, conf=0.6, verbose=False)

        # 处理results对象生成器
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 左上角和右下角坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = box.cls.cpu().numpy()  # 物体类别
                    # conf = box.conf.cpu().numpy()  # 置信度

                    # 如果物体类别是2，则说明是我们要的球的类别
                    if cls == 2:
                        # 计算中心点坐标
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2

                        # 将检测结果的中心点坐标记录添加进detect_results
                        detect_results.append({"location": {"x": x_center, "y": y_center}})

        return detect_results

    # 为每个检测目标分配初始ID
    def init_tracking_ids(self, detect_results,):
        '''
        为第一帧的检测结果分配初始跟踪ID，并初始化卡尔曼滤波器
        返回格式：
        [
            {"id": 0, "location": {"x": float, "y": float}},
            {"id": 1, "location": {"x": float, "y": float}},
            ...
        ]
        '''
        tracked_results = []
        for det in detect_results:
            x, y = det["location"]["x"], det["location"]["y"]
            # 初始化卡尔曼滤波器
            kf = self.init_kalman_filter(x, y)
            # 分配ID并存储卡尔曼滤波器
            track_id = self.next_id
            self.trackers[track_id] = {
                "kf": kf,
                "missed_frames": 0  # 连续未匹配的帧数
            }
            self.next_id += 1
            # 添加到结果
            tracked_results.append({
                "id": track_id,
                "location": {"x": x, "y": y}
            })
        return tracked_results

    # 初始化卡尔曼滤波器
    def init_kalman_filter(self, x, y):
        """
        初始化卡尔曼滤波器

        参数：
            x: 初始 x 坐标
            y: 初始 y 坐标

        滤波器配置：
            状态向量: [x, y, vx, vy]
            观测向量: [x, y]
            状态转移矩阵: 假设匀速运动模型
            观测矩阵: 只观测位置
            协方差矩阵: 初始不确定性较大
            观测噪声 (R): 默认设置为 0.1（更信任检测结果）
            过程噪声 (Q): 默认设置为 0.5（适应快速运动）

        """
        # 这里使用一个简单的卡尔曼滤波器，状态为 [x, y, vx, vy]，观测为 [x, y]
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # 状态转移矩阵 (假设匀速运动)
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # 观测矩阵
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 协方差矩阵
        kf.P *= 1000  # 初始不确定性较大
        # NOTE: 降低观测噪声（更信任检测结果）. 减少由于速度惯量太大引起的误匹配. 当前最佳：0.1
        kf.R = np.eye(2) * 0.1  # 观测噪声
        # NOTE: 提高过程噪声（更适应快速运动）. 可以适应更高的加速度方向变化. 当前最佳：0.5
        kf.Q = np.eye(4) * 0.5  # 过程噪声

        # 初始状态
        kf.x = np.array([x, y, 0, 0])

        return kf

    # 卡尔曼滤波更新步骤
    def kalman_update(self, detect_results):
        """
        卡尔曼滤波更新步骤，处理流程:

        1. 预测所有现有跟踪器的位置
        2. 使用匈牙利算法进行数据关联
        3. 处理三种情况：
            - 匹配成功的跟踪器：更新卡尔曼滤波器
            - 未匹配的检测结果：初始化新跟踪器
            - 未匹配的跟踪器：使用预测结果或删除（连续5帧未匹配）

        """
        # 1. 预测所有现有跟踪器的位置
        predictions = {}
        for track_id, tracker in self.trackers.items():
            kf = tracker["kf"]
            kf.predict()
            predictions[track_id] = kf.x[:2]  # 只取位置部分

        # 2. 数据关联：将预测位置与检测结果匹配
        matched_pairs = self.match_detections_to_predictions(detect_results, predictions)

        # 3. 处理匹配结果
        tracked_results = []
        used_det_indices = set()
        used_track_ids = set()

        # 更新匹配成功的跟踪器
        for track_id, det_idx in matched_pairs.items():
            det = detect_results[det_idx]
            x, y = det["location"]["x"], det["location"]["y"]

            # 更新卡尔曼滤波器
            kf = self.trackers[track_id]["kf"]
            kf.update(np.array([x, y]))

            # 重置未匹配帧计数器
            self.trackers[track_id]["missed_frames"] = 0

            # 添加到结果
            tracked_results.append({
                "id": track_id,
                "location": {"x": float(kf.x[0]), "y": float(kf.x[1])}
            })

            used_det_indices.add(det_idx)
            used_track_ids.add(track_id)

        # 处理未匹配的检测结果（新对象）
        for i, det in enumerate(detect_results):
            if i not in used_det_indices:
                print(f"[Tracker] 处理新加入的对象...")
                x, y = det["location"]["x"], det["location"]["y"]
                # 初始化新的卡尔曼滤波器
                kf = self.init_kalman_filter(x, y)
                track_id = self.next_id
                self.trackers[track_id] = {
                    "kf": kf,
                    "missed_frames": 0
                }
                self.next_id += 1
                tracked_results.append({
                    "id": track_id,
                    "location": {"x": x, "y": y}
                })

        # 处理未匹配的跟踪器（丢失的对象）
        for track_id in list(self.trackers.keys()):
            if track_id not in used_track_ids:
                print(f"[Tracker] 处理丢失的对象...")
                self.trackers[track_id]["missed_frames"] += 1
                # 如果连续丢失多帧，则删除该跟踪器
                if self.trackers[track_id]["missed_frames"] > 5:  # 阈值可以根据需要调整
                    del self.trackers[track_id]
                else:
                    # 使用预测结果作为输出
                    kf = self.trackers[track_id]["kf"]
                    tracked_results.append({
                        "id": track_id,
                        "location": {"x": float(kf.x[0]), "y": float(kf.x[1])}
                    })

        return tracked_results

    # 使用匈牙利算法进行数据关联
    def match_detections_to_predictions(self, detect_results, predictions):
        """
        使用匈牙利算法进行数据关联，处理流程：

        1. 构建成本矩阵（基于欧氏距离）
        2. 使用匈牙利算法找到最优匹配
        3. 应用距离阈值（默认40像素）过滤不良匹配

        """
        if not predictions or not detect_results:
            return {}

        # 构建成本矩阵（使用欧氏距离）
        cost_matrix = np.zeros((len(predictions), len(detect_results)))
        track_ids = list(predictions.keys())

        for i, track_id in enumerate(track_ids):
            pred_pos = predictions[track_id]
            for j, det in enumerate(detect_results):
                det_pos = np.array([det["location"]["x"], det["location"]["y"]])
                cost_matrix[i, j] = np.linalg.norm(pred_pos - det_pos)

        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 构建匹配结果字典 {track_id: detection_index}
        matched_pairs = {}
        for i, j in zip(row_ind, col_ind):
            # 可以设置一个最大距离阈值来过滤不良匹配
            # NOTE：根据场景缩小阈值（减少远距离错误关联），小于30会产生额外的丢失， 当前最佳：40
            if cost_matrix[i, j] < 40:  # 阈值可以根据需要调整
                matched_pairs[track_ids[i]] = j

        return matched_pairs

    def clear_history_trajectory(self):
        """清除所有历史轨迹，包括父类和该子类的缓存"""
        super().clear_history_trajectory()
        self.trackers = {}
        self.next_id = 1



































