import cv2
import os

def create_video_from_images(image_folder, output_video_file, fps=30):
    """
    从图像文件夹中生成视频。
    参数:
    image_folder: 包含图像的文件夹路径。
    output_video_file: 输出视频的文件名。
    fps: 视频的帧率。
    """
    # 获取图像列表
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]
    # 读取第一张图片以确定视频尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # 定义视频编码器和创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID'
    video = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    video.release()  # 释放资源

# 指定输出文件夹
output_folder = "/home/ly/Data/CV/output_keyframes_Linear_video/"

# 生成视频的放大倍数和对应文件夹
magnification_folders = {
    5: "/home/ly/Data/CV/output_keyframes_Linear/magnified_5x/",
    10: "/home/ly/Data/CV/output_keyframes_Linear/magnified_10x/",
    20: "/home/ly/Data/CV/output_keyframes_Linear/magnified_20x/",
    40: "/home/ly/Data/CV/output_keyframes_Linear/magnified_40x/"
}

# 为每个放大倍数生成视频文件
for factor, folder in magnification_folders.items():
    output_video_file = os.path.join(output_folder, f"output_video_{factor}x.mp4")
    create_video_from_images(folder, output_video_file)