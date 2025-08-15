'''
加载数据集：
test_dataset
    ├──video_0001
    |   └──images                     # 存放视频帧的文件夹
    |       ├──1754727254941.jpg
    |       ├──1754727254969.jpg
    |       ...
    |       └──1754727255518.jpg
    ├──video_0002
	...

将数据集中的图像帧重命名为：
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

import os

# 数据集根目录
root_dir = "test_dataset2"

for video_name in os.listdir(root_dir):
    video_path = os.path.join(root_dir, video_name, "images")
    if not os.path.isdir(video_path):
        continue

    # 获取当前视频的所有帧文件并按文件名排序
    frame_files = sorted(os.listdir(video_path))

    # 遍历并重命名
    for idx, old_name in enumerate(frame_files, start=1):
        old_path = os.path.join(video_path, old_name)
        new_name = f"frame_{idx:04d}.jpg"  # 4 位数字，前面补零
        new_path = os.path.join(video_path, new_name)

        os.rename(old_path, new_path)

    print(f"{video_name} 重命名完成，共 {len(frame_files)} 帧")

print("所有视频帧重命名完成 ✅")