'''
这里实现基础追踪器类，定义所有追踪器必须实现的方法
'''

class BaseTracker():
    def __init__(self):
        # 用于存放所有原始帧数据的列表
        self.frames = []
        # 用于存放所有检测标注的字典
        self.detect_annotations = {}  # {"frame_idx": detect_annotation} key为帧索引，detect_annotation为标注字典
        # 用于存放每一帧预测结果的字典
        self.predict_results = {}  # {"frame_idx": predict_result}  key为帧索引，predict_result为预测结果字典

    def clear_history_trajectory(self):
        '''
        清除所有历史轨迹
        '''
        self.frames = []
        self.detect_annotations = {}
        self.predict_results = {}
        print(f"[Tracker] 已清除所有历史轨迹缓存，可开始新的预测")

    def get_history_trajectory(self):
        '''
        获取历史轨迹
        整合历史中条件帧（输入标注）和非条件帧（模型预测）的信息，
        并统一格式返回。
        '''
        raise NotImplementedError("由子类实现不同获取历史轨迹方法")


    def add_condition_frame(self, frame, detect_annotation):
        '''
        当前帧为条件帧，仅记录条件信息不做预测：
        - 接受初始的图像帧（ndarray）
        - 接受初始标注信息（dict）
        '''
        # 将条件帧信息保存在 self.frames 中
        self.frames.append(frame)
        # 在对应索引上添加啊检测标注信息
        frame_idx = len(self.frames)-1
        self.detect_annotations[frame_idx] = detect_annotation


    def predict_next_frame(self, frame):
        '''
        当前帧为非条件帧，基于旧的帧预测新的帧：
        - 接受新一帧的图像帧（ndarray）

        1.模型基于旧的帧预测新的帧
        2.将新预测帧记录在self.predict_results中，格式如下:
            [
                {"id":17, "location": {"x": 160.52359, "y": 119.27372}},
                {"id":13, "location": {"x": 119.11305, "y": 279.63235}},
                ...
            ]
        3.并返回新帧预测结果
        '''
        raise NotImplementedError("由子类实现不同Tracker的预测新帧方法")