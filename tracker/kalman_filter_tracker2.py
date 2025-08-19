'''
这里实现kalman滤波追踪器（改进）
使用基于黑白摄像头模型的球的YOLO目标检测（不区分球类别），再使用kalman滤波及匹配算法的追踪
来将第一帧的球类别追踪到后续的每一帧上。

具体地，YOLO 检测器定位球体中心坐标，卡尔曼滤波基于匀速运动模型预测目标轨迹并校正观测噪声，
匈牙利算法（配合欧氏距离阈值）完成检测框与预测轨迹的最优匹配。
针对新出现目标、短暂遮挡和轨迹丢失等情况，采用 ID 自增分配、预测维持和连续丢帧剔除等策略，确保跟踪结果的连续性和鲁棒性。

我们针对匹配算法进行了一系列改进，算法核心逻辑是：

1. 初始化阶段为每个检测目标分配ID并建立卡尔曼滤波器，滤波器状态包含位置和速度信息；
2. 在更新阶段，首先检测目标运动状态变化（静止到运动的转变），使用双层匹配策略；
    - 对于由静刚转为运动的帧，重置相关滤波器的速度状态，仅基于位置距离进行简单匹配；
    - 对于非由静转动的正常帧，采用位置距离+速度惩罚项的综合成本矩阵；
        - 分阶段匹配优先处理低速目标，
        - 再处理高速目标；
3. 处理三种情况：匹配成功则更新滤波器，未匹配检测初始化新跟踪器，连续5帧未匹配则删除跟踪器。


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

class KalmanFilterTracker2(BaseTracker):
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
        卡尔曼滤波更新步骤，
        改进处理方法，当发现已有的球从静止到运动时，移除全部球的速度惯量，仅按照位置坐标进行匹配

        处理流程:
        1. 检测所有跟踪器的运动状态
        2. 预测所有现有跟踪器的位置
        3. 使用匈牙利算法进行数据关联
        4. 处理三种情况：
            - 匹配成功的跟踪器：更新卡尔曼滤波器
            - 未匹配的检测结果：初始化新跟踪器
            - 未匹配的跟踪器：使用预测结果或删除（连续5帧未匹配）

        """
        # 1. 检测所有跟踪器的运动状态
        simple_match_trackers = set()
        for track_id, tracker in self.trackers.items():
            kf = tracker["kf"]
            speed = np.linalg.norm(kf.x[2:4])  # 计算当前速度大小
            tracker["last_speed"] = getattr(tracker, "last_speed", 0)
            tracker["is_moving"] = speed > 2.0  # 速度阈值设为2.0像素/帧

            # 检测从静止到运动的转变
            if not tracker.get("was_moving", False) and tracker["is_moving"]:
                # print(f"[Tracker] 检测到球 {track_id} 从静止转为运动，重置速度状态")
                # 重置速度状态为0
                kf.x[2:4] = 0
                # 可选：增大过程噪声以适应突然运动
                # NOTE: 与上文初始化过程噪声的系数保持一致. 当前最佳：0.5
                kf.Q[2:4, 2:4] = np.eye(2) * 0.5

            tracker["was_moving"] = tracker["is_moving"]

        # 2. 预测所有现有跟踪器的位置
        predictions = {}
        for track_id, tracker in self.trackers.items():
            kf = tracker["kf"]
            kf.predict()
            predictions[track_id] = {
                "position": {
                    "x": float(kf.x[0]),
                    "y": float(kf.x[1])
                },
                "velocity": kf.x[2:4],
                "simple_match": track_id in simple_match_trackers
            }

        # 3. 数据关联：将预测位置与检测结果匹配
        matched_pairs = self.match_detections_to_predictions(detect_results, predictions)

        # 4. 处理匹配结果
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
        使用匈牙利算法进行数据关联，
        - simple_tracks 当发现已有的球从静止到运动时，移除全部球的速度惯量，仅按照位置坐标进行简单匹配
        - normal_tracks 正常匹配使用改进数据关联方法，优先保证低速目标的ID连续性，处理流程：
            1. 构建双层成本矩阵（位置距离 + 速度惩罚项）
            2. 分阶段匹配：
                - 第一阶段：优先匹配低速目标（位置变化小的）
                - 第二阶段：匹配剩余高速目标
            3. 应用距离阈值（默认40像素）过滤不良匹配

        """
        if not predictions or not detect_results:
            return {}

        # 获取跟踪器状态信息
        track_info = {
            track_id: {
                "position": np.array([predictions[track_id]["position"]["x"],
                                predictions[track_id]["position"]["y"]]),  # 转换为numpy数组
                "speed": np.linalg.norm(self.trackers[track_id]["kf"].x[2:4]),
                "is_new_moving": self.check_new_moving(track_id)
            }
            for track_id in predictions.keys()
        }

        # 分离需要简单匹配和正常匹配的跟踪器
        simple_tracks = {tid: info for tid, info in track_info.items()
                         if info["is_new_moving"]}
        normal_tracks = {tid: info for tid, info in track_info.items()
                         if not info["is_new_moving"]}

        # simple tracks -----------------------------------------------------------------
        # 先处理需要简单匹配的跟踪器
        simple_matches = {}
        if simple_tracks:
            # 构建纯位置距离矩阵
            cost_matrix = np.zeros((len(simple_tracks), len(detect_results)))
            track_ids = list(simple_tracks.keys())

            for i, track_id in enumerate(track_ids):
                pos = simple_tracks[track_id]["position"]
                for j, det in enumerate(detect_results):
                    det_pos = np.array([det["location"]["x"], det["location"]["y"]])
                    cost_matrix[i, j] = np.linalg.norm(pos - det_pos)

            # 匈牙利算法匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                # NOTE：
                if cost_matrix[i, j] < 50:  # 保持距离阈值
                    simple_matches[track_ids[i]] = j

        # normal_tracks -----------------------------------------------------------------
        normal_matches = {}
        if normal_tracks and (len(simple_matches) < len(detect_results)):
            remaining_det_indices = [j for j in range(len(detect_results))
                                     if j not in simple_matches.values()]

            # 构建增强的成本矩阵（加入速度惩罚项）
            cost_matrix = np.zeros((len(normal_tracks), len(remaining_det_indices)))
            track_ids = list(normal_tracks.keys())

            for i, track_id in enumerate(track_ids):
                info = normal_tracks[track_id]

                for j, det_idx  in enumerate(remaining_det_indices):
                    det = detect_results[det_idx]  # 这里获取实际的检测结果
                    det_pos = np.array([det["location"]["x"], det["location"]["y"]])
                    # 基础成本：位置距离
                    position_cost = np.linalg.norm(info["position"] - det_pos)
                    # 速度惩罚项：速度越快惩罚越大（权重可调）
                    # NOTE：没调整过这个参数，暂不清楚起到什么作用，当前最佳：0.3
                    speed_penalty = info["speed"] * 0.3  # 速度权重系数

                    # 总成本 = 位置距离 + 速度惩罚
                    cost_matrix[i, j] = position_cost + speed_penalty

            # 分阶段匹配策略
            matched_pairs = {}
            used_det_indices = set()

            # 第一阶段：优先匹配低速目标（速度 < 阈值）
            # NOTE：用于区分高速和低俗目标的阈值（主要能区分移动和静止的目标就行），当前最佳：5.0
            speed_threshold = 5.0  # 速度阈值，可调整
            slow_indices = [i for i, tid in enumerate(track_ids)
                            if normal_tracks[tid]["speed"] < speed_threshold]

            if slow_indices:
                # 提取低速目标的子成本矩阵
                sub_matrix = cost_matrix[slow_indices, :]
                row_sub, col_sub = linear_sum_assignment(sub_matrix)

                # 处理匹配结果
                for r, c in zip(row_sub, col_sub):
                    original_row = slow_indices[r]
                    # NOTE：低速目标的匹配距离阈值，当前最佳：40
                    if cost_matrix[original_row, c] < 40:  # 保持距离阈值
                        matched_pairs[track_ids[original_row]] = remaining_det_indices[c]
                        used_det_indices.add(c)

            # 第二阶段：匹配剩余目标（包括高速目标）
            remaining_dets_stage2 = [c for c in range(len(remaining_det_indices))
                                     if c not in used_det_indices]
            remaining_tracks = [i for i in range(len(track_ids))
                                if track_ids[i] not in matched_pairs]

            if remaining_dets_stage2 and remaining_tracks:
                # 提取剩余目标的子成本矩阵
                sub_matrix = cost_matrix[np.ix_(remaining_tracks, remaining_dets_stage2)]
                row_sub, col_sub = linear_sum_assignment(sub_matrix)

                # 处理匹配结果
                for r, c in zip(row_sub, col_sub):
                    original_row = remaining_tracks[r]
                    original_col = remaining_dets_stage2[c]


                    # NOTE：根据场景缩小高速目标的阈值（减少远距离错误关联），小于30会产生额外的丢失， 当前最佳：60
                    if cost_matrix[original_row, original_col] < 60:  # 阈值可以根据需要调整
                        matched_pairs[track_ids[original_row]] = remaining_det_indices[original_col]

            normal_matches = {tid: remaining_det_indices[j] for tid, j in matched_pairs.items()}

        return {**simple_matches, **normal_matches}

    def check_new_moving(self, track_id):
        """检查是否是从静止到运动的状态变化"""
        tracker = self.trackers[track_id]
        speed = np.linalg.norm(tracker["kf"].x[2:4])

        # 当前帧运动状态
        is_moving = speed > 2.0  # 运动阈值

        # 获取上一帧状态（默认False）
        was_moving = getattr(tracker, "was_moving", False)

        # 更新状态记录
        tracker["was_moving"] = is_moving

        # 返回是否是新开始运动
        return not was_moving and is_moving

    def clear_history_trajectory(self):
        """清除所有历史轨迹，包括父类和该子类的缓存"""
        super().clear_history_trajectory()
        self.trackers = {}
        self.next_id = 1



































