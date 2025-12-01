# 定义蛇形扫描函数
def snake_scan_h(real_data):
    batch_size, channels, H, W = real_data.shape
    L = H * W

    # 初始化蛇形扫描的索引
    o1 = []
    d1 = []
    o1_inverse = [-1 for _ in range(L)]
    i, j = 0, 0
    j_d = "right"

    # 按照蛇形扫描的方式生成索引
    while i < H:
        idx = i * W + j
        o1_inverse[idx] = len(o1)
        o1.append(idx)
        if j_d == "right":
            if j < W - 1:
                j = j + 1
                d1.append(1)  # 记录方向：1表示右
            else:
                i = i + 1
                d1.append(4)  # 记录方向：4表示下
                j_d = "left"
        else:
            if j > 0:
                j = j - 1
                d1.append(2)  # 记录方向：2表示左
            else:
                i = i + 1
                d1.append(4)  # 记录方向：4表示下
                j_d = "right"

    # 用 0 代替第一个方向，因为开始时没有方向
    d1 = [0] + d1[:-1]

    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维
    # print(f"展平后的形状: {real_data_flattened.shape}")  # 输出应为 [4, 384, 196]

    # 根据蛇形扫描的索引 o1 重新排列数据
    real_data_reordered = real_data_flattened[:, :, o1]  # 逐通道重新排序
    # print(f"重新排序后的形状: {real_data_reordered.shape}")  # 输出应为 [4, 384, 196]
    c=[]
    c.append(real_data_reordered)
    c.append(o1_inverse)
    c.append(d1)
    return c


import torch
import timeit






# 定义反向蛇形扫描函数
def snake_scan_h_flip(real_data):
    batch_size, channels, H, W = real_data.shape
    L = H * W

    # 初始化反向蛇形扫描的索引
    o2 = []
    d2 = []
    o2_inverse = [-1 for _ in range(L)]

    # 设置起始位置
    if H % 2 == 1:
        i, j = H - 1, W - 1
        j_d = "left"
    else:
        i, j = H - 1, 0
        j_d = "right"

    # 按照反向蛇形扫描的方式生成索引
    while i > -1:
        idx = i * W + j
        print("woailyy")
        # o2_inverse[idx] = len(o2)
        o2_inverse[len(o2)] = idx  # 记录逆向索引
        o2.append(idx)
        if j_d == "right":
            if j < W - 1:
                j = j + 1
                d2.append(1)  # 记录方向：1表示右
            else:
                i = i - 1
                d2.append(3)  # 记录方向：3表示上
                j_d = "left"
        else:
            if j > 0:
                j = j - 1
                d2.append(2)  # 记录方向：2表示左
            else:
                i = i - 1
                d2.append(3)  # 记录方向：3表示上
                j_d = "right"

    # 用 0 代替第一个方向，因为开始时没有方向
    d2 = [0] + d2[:-1]

    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维

    # 根据反向蛇形扫描的索引 o2 重新排列数据
    real_data_reordered = real_data_flattened[:, :, o2]  # 逐通道重新排序

    # 将结果放入列表中返回
    result = []
    result.append(real_data_reordered)
    result.append(o2_inverse)
    result.append(d2)

    return result

# 定义垂直方向的蛇形扫描函数
def snake_scan_v(real_data):
    batch_size, channels, H, W = real_data.shape
    L = H * W

    # 初始化垂直方向蛇形扫描的索引
    o3 = []
    d3 = []
    o3_inverse = [-1 for _ in range(L)]

    # 设置起始位置
    i, j = 0, 0
    i_d = "down"

    # 按照垂直方向蛇形扫描的方式生成索引
    while j < W:
        assert i_d in ["down", "up"]
        idx = i * W + j
        # o3_inverse[idx] = len(o3)
        o3_inverse[len(o3)] = idx
        o3.append(idx)
        if i_d == "down":
            if i < H - 1:
                i = i + 1
                d3.append(4)  # 记录方向：4表示下
            else:
                j = j + 1
                d3.append(1)  # 记录方向：1表示右
                i_d = "up"
        else:
            if i > 0:
                i = i - 1
                d3.append(3)  # 记录方向：3表示上
            else:
                j = j + 1
                d3.append(1)  # 记录方向：1表示右
                i_d = "down"

    # 用 0 代替第一个方向，因为开始时没有方向
    d3 = [0] + d3[:-1]

    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维

    # 根据垂直方向蛇形扫描的索引 o3 重新排列数据
    real_data_reordered = real_data_flattened[:, :, o3]  # 逐通道重新排序

    # 将结果放入列表中返回
    result = []
    result.append(real_data_reordered)
    result.append(o3_inverse)
    result.append(d3)

    return result


# 定义从右到左的垂直蛇形扫描函数
def snake_scan_v_flip(real_data):
    batch_size, channels, H, W = real_data.shape
    L = H * W

    # 初始化蛇形扫描的索引
    o4 = []
    d4 = []
    o4_inverse = [-1 for _ in range(L)]

    # 设置起始位置
    if W % 2 == 1:
        i, j = H - 1, W - 1
        i_d = "up"
    else:
        i, j = 0, W - 1
        i_d = "down"

    # 按照从右到左的垂直蛇形扫描方式生成索引
    while j > -1:
        assert i_d in ["down", "up"]
        idx = i * W + j
        # o4_inverse[idx] = len(o4)
        o4_inverse[len(o4)] = idx
        o4.append(idx)
        if i_d == "down":
            if i < H - 1:
                i = i + 1
                d4.append(4)  # 记录方向：4表示下
            else:
                j = j - 1
                d4.append(2)  # 记录方向：2表示左
                i_d = "up"
        else:
            if i > 0:
                i = i - 1
                d4.append(3)  # 记录方向：3表示上
            else:
                j = j - 1
                d4.append(2)  # 记录方向：2表示左
                i_d = "down"

    # 用 0 代替第一个方向，因为开始时没有方向
    d4 = [0] + d4[:-1]

    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维

    # 根据蛇形扫描的索引 o4 重新排列数据
    real_data_reordered = real_data_flattened[:, :, o4]  # 逐通道重新排序

    # 将结果放入列表中返回
    result = []
    result.append(real_data_reordered)
    result.append(o4_inverse)
    result.append(d4)

    return result

import torch

# 创建一个简单的 4x4 二维矩阵用于测试
H, W = 4, 4
original_data = torch.arange(1, H * W + 1).reshape(1, 1, H, W).float()
print("Original Data:\n", original_data)

# 调用水平反向蛇形扫描的函数
snake_result_h_flip = snake_scan_v_flip(original_data)

# 提取重排后的数据、反向索引和方向索引
reordered_data_h_flip = snake_result_h_flip[0]
o2_inverse = snake_result_h_flip[1]
d2 = snake_result_h_flip[2]

# 打印经过 h_flip 方向蛇形扫描重排后的数据
print("Reordered Data (h_flip):\n", reordered_data_h_flip)

# 使用反向索引还原数据
# 由于 `o2_inverse` 是记录的逆向索引，可以用于还原数据顺序
reconstructed_data_h_flip = torch.zeros_like(original_data.flatten(2))
reconstructed_data_h_flip[:, :, o2_inverse] = reordered_data_h_flip  # 根据反向索引还原
reconstructed_data_h_flip = reconstructed_data_h_flip.reshape(1, 1, H, W)  # 调整为原始 H x W 形状

# 打印还原的二维矩阵
print("Reconstructed Data (h_flip):\n", reconstructed_data_h_flip)

# 检查还原数据是否与原始数据相同
if torch.equal(original_data, reconstructed_data_h_flip):
    print("Test Passed: The reconstructed data matches the original data.")
else:
    print("Test Failed: The reconstructed data does not match the original data.")
