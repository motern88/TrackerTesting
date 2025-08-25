'''
暂未实现DetSAM2的追踪器，DetSAM2不太好拆成这种形式的tracker，还是简易直接跑DetSAM2中实现的video_processor较好

'''
from tracker.base_tracker import BaseTracker

# 这里加载DetSAM2的VideoProcessor，假设



class DetSAM2Tracker(BaseTracker):
    '''

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
        # 实例化Det-SAM2的视频处理器
        video_processor = VideoProcessor(
            output_dir=output_dir,
            sam2_checkpoint=sam2_checkpoint,
            model_cfg=model_cfg,
            detect_model_weights=detect_model_weights,
            # load_inference_state_path=load_inference_state_path,  # 不传或传None,则不预加载内存库
            # save_inference_state_path=save_inference_state_path,  # 不传或传None,则不保存内存库
        )