# 计算机视觉导论期末project

## 视频的基本情况

- **视频时长**：视频总时长为12小时00分34.91秒。
- 视频流信息：
  - **视频流**：包含一个视频流，编码为`H.264`，分辨率为 `1280x720（720p）`，帧率为 `29.97 FPS`。
  - **音频流**：包含一个音频流，编码为 `AAC`，采样率为 `44100 Hz`，双声道。

对于目前资源有限，而视频内容过于庞大，因此需要处理视频。目前的处理思路是**使用`ffmpeg`提出关键帧图片，经过后续算法处理之后将图片拼接为视频输出**

## 视频处理

### 提取关键帧图片

#### 命令行输入

```bash
ffmpeg -i /home/ly/Data/CV/video_for_motion_mag.mp4 -vf "select='eq(pict_type,PICT_TYPE_I)'" -vsync vfr "/home/ly/Data/CV/output_keyframes/output_keyframes_%04d.jpg"
```

 `ffmpeg` 命令详解如下：

- **`-i /home/ly/Data/CV/video_for_motion_mag.mp4`**：这指定了输入文件的路径。
- **`-vf "select='eq(pict_type,PICT_TYPE_I)'"`**：这个视频过滤器参数 (`-vf`) 用于选择类型为 I 的帧（关键帧）。`pict_type` 是` ffmpeg `中用来区分不同类型帧的内部标记，`PICT_TYPE_I` 表示关键帧。
- **`-vsync vfr`**：这个参数设置视频同步模式为变帧率（Variable Frame Rate），这样可以确保只有在需要时才输出帧，适用于输出数量不定的关键帧。
- **`"/home/ly/Data/CV/output_keyframes/output_keyframes_%04d.jpg"`**：这指定了输出文件的保存路径和文件名模式。`%04d` 表示数字以四位数格式递增，用于每个输出文件的命名。

#### 输出结果

```bash
#0/image2 @ 0x55b17f52cd80] video:429474kB audio:0kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: unknown
frame= 8640 fps= 20 q=1.6 Lsize=N/A time=12:00:28.01 bitrate=N/A speed= 101x   
```

- **`video:429474kB`**: 表示输出的视频数据总量为`429474kB`（即大约`429MB`）。这里指的是所有生成的JPEG图像文件的总大小。

- **`frame= 8640`**: 共提取了**8640帧作为关键帧**。这意味着在整个12小时的视频中，共有8640个I帧被识别并保存为JPEG图片。
- **`fps= 20`**: 处理速度保持在每秒20帧，这表示 `ffmpeg` 在提取关键帧时的处理速率。
- **`q=1.6`**: 指定JPEG输出的质量因子，其中1.0是最佳质量（无损压缩）。**`1.6`非常接近最佳质量，表明输出的图像质量非常高。**

##  **Eulerian 方法中的线性方法**

```python
def magnify_motion(images, magnification_factor):
    """
    放大图像序列中的运动。该函数通过计算连续两帧之间的差异，并将差异乘以一个放大系数，
    然后将增强的差异加回原始图像中，从而实现运动的放大效果。
    """
    output_images = []  # 创建一个空列表，用于存储处理后的图像
    prev_image = images[0]  # 初始化前一帧图像为序列的第一帧
    for i in range(1, len(images)):
        current_image = images[i]  # 获取当前处理的帧
        frame_diff = cv2.absdiff(current_image, prev_image)  # 计算当前帧与前一帧之间的差异
        magnified_diff = cv2.multiply(frame_diff, np.array([magnification_factor], dtype=np.uint8))  # 将差异乘以放大系数
        enhanced_image = cv2.add(current_image, magnified_diff)  # 将放大的差异添加回当前帧
        output_images.append(enhanced_image)  # 将处理后的图像添加到输出列表中
        prev_image = current_image  # 更新前一帧图像为当前帧，为下一次循环做准备

    return output_images  # 返回处理后的图像序列
```

将生成的图片转化为视频输出

<img src="Experience\Linear 01.jpg" style="zoom:50%;" />

效果评估：生成的最终视频中，20、40倍放大的视频中**出现大量蓝点**的现象，影响最后的效果，可能的原因如下：

1. **噪声放大**：放大系数越大，视频中原本微不足道的噪声也会被显著放大。这种情况尤其在低光环境或使用低质量摄像头拍摄的视频中更为常见。如果原始视频中存在轻微的颜色偏差或者噪点，在放大过程中这些细微的噪声也会被放大，表现为明显的色点。

2. **压缩和编码问题**：视频的压缩和编码过程可能会引入块状噪声或色彩失真，尤其是在高放大倍数下，这些问题会变得更加明显。这种情况下，编码器可能会在尝试压缩带有微小变化的视频时产生错误，从而引入额外的噪声。

3. **处理算法的限制**：如果放大算法主要是线性放大像素差异，可能不太能有效区分真正的运动和噪声。因此，在放大过程中，不仅真正的运动被放大，噪声同样被放大，导致视频质量下降。

综上所述，选择可以采用新的算法尝试。

## Eulerian 方法中的相位基方法

### 第一次尝试

```python
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


def phase_magnify(channel, magnification_factor, levels):
    """相位放大单个颜色通道。"""
    g_pyr = build_gaussian_pyramid(channel, levels)
    l_pyr = laplacian_from_gaussian(g_pyr)
    
    # 处理每一层的相位
    for i in range(len(l_pyr)):
        complex_layer = np.fft.fft2(l_pyr[i].astype(np.float32))
        magnitude = np.abs(complex_layer)
        phase = np.angle(complex_layer)
        # 直接计算放大的相位
        magnified_phase = phase * (1 + magnification_factor)  # 放大相位
        new_complex_layer = magnitude * np.exp(1j * magnified_phase)
        l_pyr[i] = np.fft.ifft2(new_complex_layer).real

    return reconstruct_from_laplacian_pyramid(l_pyr)

def phase_based_motion_magnification(images, magnification_factor, levels=3):
    output_images = []
    # 为每幅图像的每个颜色通道应用相位放大并重建
    for img in tqdm(images, desc="处理图像"):
        channels = cv2.split(img)
        magnified_channels = []
        for channel in channels:
            magnified_channel = phase_magnify(channel, magnification_factor, levels)
            magnified_channels.append(np.clip(magnified_channel, 0, 255).astype(np.uint8))
        magnified_image = cv2.merge(magnified_channels)
        output_images.append(magnified_image)
    return output_images
```

放大五倍的某一帧结果照片如下：

<img src="Experience\fail 01.png" style="zoom:50%;" />

图像的失真可能是**由于FFT和IFFT处理过程中某些步骤不正确或相位放大过程中出现的问题**。这可能导致颜色变得异常，尤其是在频域处理后将复数结果转换回实数图像时。

### 第二次尝试：减小相位放大因子

```python
def phase_magnify(channel, magnification_factor, levels):
    g_pyr = build_gaussian_pyramid(channel, levels)
    l_pyr = laplacian_from_gaussian(g_pyr)
    
    for i in range(len(l_pyr)):
        complex_layer = np.fft.fft2(l_pyr[i].astype(np.float32))
        magnitude = np.abs(complex_layer)
        phase = np.angle(complex_layer)
        magnified_phase = phase * (1 + magnification_factor / 10)  # 减小放大因子
        new_complex_layer = magnitude * np.exp(1j * magnified_phase)
        l_pyr[i] = np.fft.ifft2(new_complex_layer).real
        l_pyr[i] = np.clip(l_pyr[i], 0, 255)  # 确保值在有效范围内

    return reconstruct_from_laplacian_pyramid(l_pyr)
```

放大五倍的某一帧结果照片如下，效果比刚刚要更好，但仍然需要优化：

<img src="Experience\fail 02.jpg" style="zoom:50%;" />

### 第三次尝试：自适应相位放大算法 

```python
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
```

放大五倍的某一帧结果照片如下：

<img src="Experience\Phase-based\output_5x_0007_phase（无去噪）.jpg" style="zoom:50%;" />

在这次尝试中我又对比了加入了**双边滤波或者非局部均值去噪**的图像结果

```python
img = cv2.bilateralFilter(img, d=-1, sigmaColor=75, sigmaSpace=15) # 使用双边滤波进行噪声抑制
img= cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21) # 非局部均值去噪
```

<img src="Experience\Phase-based\combined_image.jpg" style="zoom:25%;" />

使用双边滤波或者非局部均值去噪之后图像的**清晰度明显降低**，因此实际应用中我并未使用。

## 结论

1. 视频处理：碍于`12h`视频过大，再加上自己资源有限，故采用抽取关键帧的方式便于处理。
2. 采用**线性方法和基于相位**的方法（与原本的欧拉方式有出入），没有采用基于深度学习的方法主要是碍于自身资源有限。以下是一些方法实践中产生的对比：
   1. 线性方法
      - **计算效率高**：因为主要涉及基本的图像处理操作，所以处理速度通常较快。我统计了一下8640帧总共`10m 16.1s`完成。
      - **噪声敏感性强**：线性方法直接放大所有动态，包括噪声，就像我前面所说，导致放大的图像中噪声过多，出现了**大量的蓝点**。
   2. 相位方法
      - **鲁棒性高**：相位方法通过操作图像的相位信息，对噪声有更好的抵抗力，放大效果更加平滑和自然，观看视频可以看到线性方法产生的猫动作比较卡顿，而**相位方法更加顺畅**。
      - **适用于复杂运动**：这种方法能够有效处理复杂或微小的运动放大，尤其是仔细观看**猫的毛发变化时**。
      - **计算复杂度高，效率低**：相较于线性方法，基于相位的方法需要进行更多的计算，如傅里叶变换和金字塔构建，因此处理时间更长。
        - 处理5倍放大时间总共`1:21:30`，处理10倍放大时间总共`1:21:12`，处理20倍放大时间总共`1:21:21`，处理40倍放大时间总共`1:21:59`，时间总计`5h26min2s`，与线性方法相差极大。
3. 在运动放大过程中可以使用一些去噪方法，但是我在试用**双边滤波或者非局部均值去噪**时发现会损失清晰度，尤其是**损失边缘像素**
   - **边缘“光晕”效应**：在高对比度边缘附近，双边滤波可能产生光晕效应，即边缘周围出现不自然的模糊区域，这可能影响图像的视觉质量。
   - **边缘模糊**：虽然非局部均值去噪在保持图像结构方面通常优于传统局部方法，但不恰当的参数选择仍可能导致边缘和细节的模糊。

**注：报告由markdown撰写，故无法在其中嵌入本地视频，视频文件已经与代码共同打包上传elearning，请查收！**
