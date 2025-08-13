'''
这里实现基础追踪器类，定义所有追踪器必须实现的方法
'''

class BaseTracker():
    def __init__(self):
        # 用于存放所有原始帧数据的列表
        self.frames = []
        # 用于存放每一帧预测结果的字典
        # {"frame_idx": predict_result_list}  key为帧索引，predict_result_list为预测结果列表
        # predict_result_list = [{"id":1, "location": {"x": 160.52359, "y": 119.27372}},...]
        self.predict_results = {}

    def clear_history_trajectory(self):
        '''
        清除所有历史轨迹
        '''
        self.frames = []
        self.predict_results = {}
        print(f"[Tracker] 已清除所有历史轨迹缓存，可开始新的预测")

    def get_history_trajectory(self):
        '''
        获取历史轨迹
        '''
        raise self.predict_results

    # 需由子类实现该process_frame方法
    def process_frame(self, frame):
        '''
        1.接受新的图像帧输入，记录在self.frames中
        2.执行推理，并将将新预测帧记录在self.predict_results中，{"frame_idx": predict_result_list}
        其中 predict_result_list 格式如下:
            [
                {"id":17, "location": {"x": 160.52359, "y": 119.27372}},
                {"id":13, "location": {"x": 119.11305, "y": 279.63235}},
                ...
            ]
        '''
        # 1.接受新的帧信息并保存在 self.frames 中
        self.frames.append(frame)
        # 计算帧索引
        frame_idx = len(self.frames)-1

        # 2.执行推理并将新的推理结果记录在self.predict_results中

        raise NotImplementedError("由子类实现不同Tracker的process_frame推理方法")
