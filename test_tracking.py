'''
测试追踪部分算法的脚本

读取测试数据集，给定追踪算法开头一帧的检测结果，
追踪算法在后续的每一帧中进行追踪，
根据每一帧的检测结果比对追踪算法的预测结果
'''
import os
import re
import json
import cv2

class TestTracking():

    def __init__(self, test_dataset_path, tracker):
        '''
        数据集格式：
        test_dataset
            ├──video_0001
            |   └──images                     # 存放视频帧的文件夹
            |       ├──frame_0001.jpg
            |       ├──frame_0002.jpg
            |       ...
            |       └──frame_XXXX.jpg
            ├──video_0002
            ...
        '''
        # 测试数据集路径
        self.test_dataset_path = test_dataset_path
        # 实现的自定义的
        self.tracker = tracker

    def load_img(self,img_path):
        '''opencv加载图像，返回nd数组'''
        img = cv2.imread(img_path)
        return img

    # 测试追踪算法完整流程
    def start_test_tracking(self):
        '''
        加载数据集测试追踪算法完整流程:
        1.遍历测试数据集下每一个 video_{四位数ID} 的文件夹
        2.处理每个视频数据，从其第一帧开始进行追踪预测
        '''
        # 1.按顺序遍历测试数据集下每一个video文件夹
        video_dirs = sorted(os.listdir(self.test_dataset_path))
        for video_dir in video_dirs:
            video_path = os.path.join(self.test_dataset_path, video_dir)
            if not os.path.isdir(video_path):
                continue
            # 2.处理每一个视频数据
            print(f"[system] 正在处理 {video_dir} ...")
            self.process_video(video_path)

    # 测试脚本对每个视频样本数据的处理过程
    def process_video(self, video_path):
        '''
        在 video_path 下:
        images                # 存放视频帧的文件夹
           ├──frame_0001.jpg
           ├──frame_0002.jpg
           ...
           └──frame_XXXX.jpg

        每个样本遍历全部帧，每帧有追踪器Tracker处理并获取返回结果。
        '''
        images_dir = os.path.join(video_path, "images")  # 图像的路径

        # 按帧名排序（确保按顺序读取）
        frame_files = sorted(os.listdir(images_dir))

        frame_id = 1  # 初始化帧计数器
        for frame_file in frame_files[1:]:
            # 加载图像帧为nd数组
            frame_path = os.path.join(images_dir, frame_file)
            frame= self.load_img(frame_path)

            # 调用追踪器处理当前帧
            self.tracker.process_frame(frame)

            frame_id += 1

        # 3. 完成视频所有帧遍历后，记录预测结果，并清除Tracker轨迹
        # 一次性获取历史推理轨迹
        history_trajectory = self.tracker.get_history_trajectory()
        
        # 清除历史Tracker轨迹
        self.tracker.clear_history_trajectory()




if __name__ == "__main__":
    '''
    进行测试，根目录下运行 python test_tracking.py
    '''
    from tracker.base_tracker import BaseTracker
    from tracker.kalman_filter_tracker import KalmanFilterTracker

    # --------- 1.初始化要测试的追踪器 ---------
    tracker = KalmanFilterTracker()
    # tracker = BaseTracker()


    # --------- 2. 初始化追踪测试器 -----------
    test_tracking = TestTracking(
        test_dataset_path = "./dataset/test_dataset",
        tracker = tracker
    )
    test_tracking.start_test_tracking()


