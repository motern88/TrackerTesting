'''
这里实现kalman滤波追踪器
使用基于黑白摄像头模型的球检测（不区分球类别），再使用kalman滤波追踪来将第一帧的球类别追踪到后续的每一帧上。
'''
from base_tracker import BaseTracker
from ultralytics import YOLO

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
        super().__init__()
        '''
        继承父类初始化的缓存结构：
        1. self.frames = [] 用于存放所有原始帧数据的列表: 
        
        2. self.predict_results = {} 用于存放每一帧预测结果的字典: 
            {"frame_idx": predict_result_list}  key为帧索引，predict_result_list为预测结果列表
            predict_result_list = [{"id":1, "location": {"x": 160.52359, "y": 119.27372}},...]
        '''
        self.model = YOLO("model/train_ir_9_640.pt")  # 加载YOLO模型
        print(f"[Tracker] 初始化完成")

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


    def kalman_update(self, detections):
        """
        输入检测到的位置，用卡尔曼滤波进行预测和更新
        """
        raise NotImplementedError



































