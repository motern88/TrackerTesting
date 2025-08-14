'''
测试追踪部分算法的脚本

读取测试数据集，每一帧都交给Tracker进行追踪
输出追踪算法的预测结果
'''
import os
import re
import json
import cv2
import random

class TestTracking():

    def __init__(self, test_dataset_path, tracker, output_dir):
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

        # 可视化路径
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

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
        images
           ├──frame_0001.jpg
           ├──frame_0002.jpg
           ...
           └──frame_XXXX.jpg

        每个样本遍历全部帧，每帧有追踪器Tracker处理并获取返回结果。
        '''
        images_dir = os.path.join(video_path, "images")  # 图像的路径
        video_name = os.path.basename(video_path)  # 用文件夹名作为可视化的视频名

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
        # print(f"[system] 推理结果轨迹：\n {history_trajectory}")

        # 绘制可视化轨迹
        visualized_frames = self.visualize_trajectory(history_trajectory, self.tracker.frames)

        # 将轨迹帧保存为 MP4
        if visualized_frames:
            height, width = visualized_frames[0].shape[:2]
            save_path = os.path.join(self.output_dir, f"{video_name}.mp4")

            # 使用 mp4v 编码（H.264 可用 avc1，如果系统支持）
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 3
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

            for frame in visualized_frames:
                out.write(frame)
            out.release()

            print(f"[system] 轨迹视频已保存: {save_path}")
        else:
            print("[system][warning] 没有生成任何可视化帧，视频未保存。")

        # 清除历史Tracker轨迹(防止影响下一个视频)
        self.tracker.clear_history_trajectory()

    # 可视化历史推理轨迹
    def visualize_trajectory(self, history_trajectory, frames):
        '''
        为所有帧frames绘制历史轨迹可视化。

        - history_trajectory 为所有物体的轨迹{"frame_idx": predict_result_list}：
            {
                "0": [
                    {"id":17, "location": {"x": 160.52359, "y": 119.27372}},
                    {"id":13, "location": {"x": 119.11305, "y": 279.63235}},
                    ...
                ],
                "1": [...]
                ...
            }
        - frames 是存放所有图像帧的列表，其中元素为对应帧经过cv.imread()加载后的nd数组
        '''
        # 用于存储每个 id 的历史点坐标
        trajectory_points = {}
        # 存储每个 id 的颜色 (BGR)
        id_colors = {}

        def get_color_for_id(obj_id):
            """固定 ID 颜色"""
            if obj_id not in id_colors:
                random.seed(obj_id)  # 保证相同 ID 每次生成的颜色一致
                id_colors[obj_id] = (
                    random.randint(50, 255),  # B
                    random.randint(50, 255),  # G
                    random.randint(50, 255)  # R
                )
            return id_colors[obj_id]

        for frame_idx_str, objects in history_trajectory.items():
            frame_idx = int(frame_idx_str)
            frame = frames[frame_idx].copy()

            for obj in objects:
                obj_id = obj["id"]
                x = int(obj["location"]["x"])
                y = int(obj["location"]["y"])
                color = get_color_for_id(obj_id)

                # 保存轨迹点
                if obj_id not in trajectory_points:
                    trajectory_points[obj_id] = []
                trajectory_points[obj_id].append((x, y))

                # 画当前点
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{obj_id}", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 如果有历史点，画轨迹线
                if len(trajectory_points[obj_id]) > 1:
                    for i in range(1, len(trajectory_points[obj_id])):
                        cv2.line(frame,
                                 trajectory_points[obj_id][i - 1],
                                 trajectory_points[obj_id][i],
                                 color, 2)

            # 更新 frames（如果需要保存结果）
            frames[frame_idx] = frame

        return frames


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
        tracker = tracker,
        output_dir = "./output"
    )
    test_tracking.start_test_tracking()


