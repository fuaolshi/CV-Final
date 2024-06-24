import cv2
import numpy as np
import os
import gc
from tqdm import tqdm

def load_images_from_folder(folder, batch_size):
    """从文件夹分批加载图像。
    参数:
    folder: 存放图像的文件夹路径。
    batch_size: 每批加载的图像数量。

    返回:
    生成器，逐批返回加载的图像列表。
    """
    filenames = sorted(os.listdir(folder))
    total_batches = (len(filenames) + batch_size - 1) // batch_size  # 计算总批次
    for i in tqdm(range(0, len(filenames), batch_size), total=total_batches, desc="加载图像批次"):
        batch_images = []
        for filename in filenames[i:i+batch_size]:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                batch_images.append(img)
        yield batch_images

def magnify_motion(images, magnification_factor):
    """放大图像序列中的运动。
    参数:
    images: 输入的图像序列。
    magnification_factor: 放大系数。

    返回:
    output_images: 处理后的图像序列。
    """
    output_images = []
    prev_image = images[0]
    for i in range(1, len(images)):
        current_image = images[i]
        frame_diff = cv2.absdiff(current_image, prev_image)
        magnified_diff = cv2.multiply(frame_diff, np.array([magnification_factor], dtype=np.uint8))
        enhanced_image = cv2.add(current_image, magnified_diff)
        output_images.append(enhanced_image)
        prev_image = current_image
    return output_images

def save_images_to_folder(images, folder, magnification_factor, start_index=0):
    """将图像保存到指定文件夹。
    参数:
    images: 需要保存的图像序列。
    folder: 目标文件夹路径。
    magnification_factor: 放大系数。
    start_index: 开始保存的图像索引。
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 使用 start_index 作为初始索引
    for i, img in enumerate(tqdm(images, desc=f"保存放大系数 {magnification_factor}x 的图像"), start=start_index):
        filename = os.path.join(folder, f"output_{magnification_factor}x_{i:04d}.jpg")
        cv2.imwrite(filename, img)
    return i + 1  # 返回最后一个保存的图像索引


# 使用示例
input_folder = '/home/ly/Data/CV/output_keyframes/'
base_output_folder = '/home/ly/Data/CV/output_keyframes_Linear/'
batch_size = 800  # 可以根据实际情况调整批次大小
magnification_levels = [5, 10, 20, 40]

last_index = 0  # 初始索引为 0
for factor in magnification_levels:
    specific_output_folder = os.path.join(base_output_folder, f"magnified_{factor}x")
    for images_batch in load_images_from_folder(input_folder, batch_size):
        magnified_images = magnify_motion(images_batch, factor)
        last_index = save_images_to_folder(magnified_images, specific_output_folder, factor, last_index)
        del images_batch, magnified_images
        gc.collect()  # 进行垃圾回收
    last_index = 0  # 重置索引以适应新的放大倍数