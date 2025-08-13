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
    - TODO:获取历史轨迹
        self.get_history_trajectory()
    - 添加条件帧
        self.add_condition_frame(frame, detect_annotation)
    - TODO：预测下一帧（非条件帧）
        self.predict_next_frame(frame)

    以上部分方法由BaseTracker实现，部分方法由本子类实现。
    本子类主要实现基于YOLO检测+kalman滤波的追踪算法
    '''
    def __init__(self):
        super().__init__()
        '''
        继承父类初始化的缓存结构：
        
        1. 用于存放所有原始帧数据的列表: self.frames = []
        
        2. 用于存放所有检测标注的字典: self.detect_annotations = {}
            {"frame_idx": detect_annotation} key为帧索引，detect_annotation为标注字典
        
        3. 用于存放每一帧预测结果的字典: self.predict_results = {}
            {"frame_idx": predict_result}  key为帧索引，predict_result为预测结果字典
        '''
        self.yolo = YOLO(yolo_model_path)  # 加载YOLO模型
        print(f"[Tracker] 初始化完成")

    def get_history_trajectory(self): 
        history = []
        for idx in range(len(self.frames)):
            if idx in self.detect_annotations:
                # 将{"id":17, "location":{"bottom":132.50513,"left":160.52359,"right":175.47101,"top":119.27372}}
                # 处理成{"id":1,"location":{"x": 119.11305, "y": 279.63235}}
                # TODO
            elif idx in self.predict_results:
                # 确保是"id":1,"location":{"x": 119.11305, "y": 279.63235}}格式
                history.append(self.predict_results[idx])
        return history

    def predict_next_frame(self, frame):
        '''
        当前帧为非条件帧，基于历史帧预测并追踪球位置：
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
        # [{"location": [x,y,w,h]}, ...]

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
        黑白yolo模型进行检测
        '''
        raise NotImplementedError

    def kalman_update(self, detections):
        """
        输入检测到的位置，用卡尔曼滤波进行预测和更新
        """
        raise NotImplementedError



































