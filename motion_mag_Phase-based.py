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

def build_gaussian_pyramid(img, levels):
    """使用 OpenCV 创建高斯金字塔。"""
    pyramid = [img]
    for _ in range(levels - 1):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid

def laplacian_from_gaussian(gaussian_pyr):
    """从高斯金字塔生成拉普拉斯金字塔。"""
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        L = cv2.subtract(gaussian_pyr[i], cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size))
        laplacian_pyr.append(L)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr

def reconstruct_from_laplacian_pyramid(lpyr):
    """从拉普拉斯金字塔重建图像，确保数据类型和范围。"""
    img = lpyr[-1].astype(np.float32)  # 确保顶层是float类型
    for layer in reversed(lpyr[:-1]):
        img = cv2.pyrUp(img, dstsize=(layer.shape[1], layer.shape[0])).astype(np.float32)
        layer_float = layer.astype(np.float32)  # 确保layer也是float类型
        img = cv2.add(img, layer_float)  # 使用相同类型的数组进行加法
    img = np.clip(img, 0, 255)  # 确保图像值在0-255范围内
    return img.astype(np.uint8)  # 转换为uint8类型以便显示和保存

def adaptive_phase_amplification(magnitude, phase, base_magnification):
    """根据局部幅度信息和非线性响应来调整相位放大的程度。"""
    # 计算自适应放大因子，这里引入一个限制因子以减少放大的极端效果
    adaptive_factor = np.tanh(magnitude / np.max(magnitude) * base_magnification / 10)
    # 计算放大后的相位
    amplified_phase = phase + adaptive_factor * np.sign(phase)  # 保持原相位符号
    return amplified_phase

def phase_magnify(channel, magnification_factor, levels):
    """相位放大单个颜色通道。"""
    g_pyr = build_gaussian_pyramid(channel, levels)
    l_pyr = laplacian_from_gaussian(g_pyr)
    
    # 处理每一层的相位
    for i in range(len(l_pyr)):
        complex_layer = np.fft.fft2(l_pyr[i].astype(np.float32))
        magnitude = np.abs(complex_layer)
        phase = np.angle(complex_layer)
        magnified_phase = adaptive_phase_amplification(magnitude, phase, magnification_factor) # 使用自适应相位放大
        new_complex_layer = magnitude * np.exp(1j * magnified_phase)
        l_pyr[i] = np.fft.ifft2(new_complex_layer).real
        l_pyr[i] = np.clip(l_pyr[i], 0, 255)  # 确保值在有效范围内

    return reconstruct_from_laplacian_pyramid(l_pyr)

def phase_based_motion_magnification(images, magnification_factor, levels=3):
    output_images = []
    # 为每幅图像的每个颜色通道应用相位放大并重建
    for img in tqdm(images, desc="处理图像"):
        # img = cv2.bilateralFilter(img, d=-1, sigmaColor=75, sigmaSpace=15) # 使用双边滤波进行噪声抑制
        # img= cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) # 非局部均值去噪
        channels = cv2.split(img)
        magnified_channels = []
        for channel in channels:
            magnified_channel = phase_magnify(channel, magnification_factor, levels)
            magnified_channels.append(np.clip(magnified_channel, 0, 255).astype(np.uint8))
        magnified_image = cv2.merge(magnified_channels)
        output_images.append(magnified_image)
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
base_output_folder = "/home/ly/Data/CV/output_keyframes_Phase-based/"
batch_size = 800  # 可以根据实际情况调整批次大小
magnification_levels = [5, 10, 20, 40]

last_index = 0  # 初始索引为 0
for factor in magnification_levels:
    specific_output_folder = os.path.join(base_output_folder, f"magnified_{factor}x")
    for images_batch in load_images_from_folder(input_folder, batch_size):
        magnified_images = phase_based_motion_magnification(images_batch, factor)
        last_index = save_images_to_folder(magnified_images, specific_output_folder, factor, last_index)
        del images_batch, magnified_images
        gc.collect()  # 进行垃圾回收
    last_index = 0  # 重置索引以适应新的放大倍数