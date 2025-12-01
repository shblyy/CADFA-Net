import torch
# 初始化缓存字典
index_cache = {}


# 生成指定方向的索引并缓存
def precompute_snake_scan_indices(H, W):
    # print(H)
    L = H * W

    # 水平蛇形扫描（从左到右）
    o1, d1, o1_inverse = [], [], [-1 for _ in range(L)]
    i, j, j_d = 0, 0, "right"
    while i < H:
        idx = i * W + j
        o1_inverse[idx] = len(o1)
        # o1_inverse[len(o1)] = idx  # 记录逆向索引
        o1.append(idx)
        if j_d == "right":
            if j < W - 1:
                j += 1
                d1.append(1)
            else:
                i += 1
                d1.append(4)
                j_d = "left"
        else:
            if j > 0:
                j -= 1
                d1.append(2)
            else:
                i += 1
                d1.append(4)
                j_d = "right"
    d1 = [0] + d1[:-1]
    index_cache[(H, W, "h")] = (o1, o1_inverse, d1)

    # 反向水平蛇形扫描（从右到左）
    o2, d2, o2_inverse = [], [], [-1 for _ in range(L)]
    if H % 2 == 1:
        i, j, j_d = H - 1, W - 1, "left"
    else:
        i, j, j_d = H - 1, 0, "right"
    while i > -1:
        idx = i * W + j
        o2_inverse[idx] = len(o2)
        # o2_inverse[len(o2)] = idx  # 记录逆向索引
        o2.append(idx)
        if j_d == "right":
            if j < W - 1:
                j += 1
                d2.append(1)
            else:
                i -= 1
                d2.append(3)
                j_d = "left"
        else:
            if j > 0:
                j -= 1
                d2.append(2)
            else:
                i -= 1
                d2.append(3)
                j_d = "right"
    d2 = [0] + d2[:-1]
    index_cache[(H, W, "h_flip")] = (o2, o2_inverse, d2)

    # 垂直蛇形扫描（从上到下）
    o3, d3, o3_inverse = [], [], [-1 for _ in range(L)]
    i, j, i_d = 0, 0, "down"
    while j < W:
        idx = i * W + j
        o3_inverse[idx] = len(o3)
        # o3_inverse[len(o3)] = idx  # 记录逆向索引
        o3.append(idx)
        if i_d == "down":
            if i < H - 1:
                i += 1
                d3.append(4)
            else:
                j += 1
                d3.append(1)
                i_d = "up"
        else:
            if i > 0:
                i -= 1
                d3.append(3)
            else:
                j += 1
                d3.append(1)
                i_d = "down"
    d3 = [0] + d3[:-1]
    index_cache[(H, W, "v")] = (o3, o3_inverse, d3)

    # 反向垂直蛇形扫描（从右到左）
    o4, d4, o4_inverse = [], [], [-1 for _ in range(L)]
    if W % 2 == 1:
        i, j, i_d = H - 1, W - 1, "up"
    else:
        i, j, i_d = 0, W - 1, "down"
    while j > -1:
        idx = i * W + j
        o4_inverse[idx] = len(o4)
        # o4_inverse[len(o4)] = idx  # 记录逆向索引
        o4.append(idx)
        if i_d == "down":
            if i < H - 1:
                i += 1
                d4.append(4)
            else:
                j -= 1
                d4.append(2)
                i_d = "up"
        else:
            if i > 0:
                i -= 1
                d4.append(3)
            else:
                j -= 1
                d4.append(2)
                i_d = "down"
    d4 = [0] + d4[:-1]
    index_cache[(H, W, "v_flip")] = (o4, o4_inverse, d4)


# 预计算常用尺寸的索引
for size in [56, 28, 14, 7]:
    # print("suoyin:",size)
    precompute_snake_scan_indices(size, size)
# 使用缓存索引的函数示例
def snake_scan_with_cache(real_data, direction="h"):
    batch_size, channels, H, W = real_data.shape
    # print("real_data.shape:",real_data.shape)
    # 检查 H 和 W 是否为 tensor，如果是则转换为整数
    if isinstance(H, torch.Tensor):
        H = H.item()
    if isinstance(W, torch.Tensor):
        W = W.item()
    # print((H, W, direction))
    if (H, W, direction) in index_cache:
        # print("12wozaihuancun：",H)
        o, o_inverse, d = index_cache[(H, W, direction)]
    else:
        # 如果没有缓存，生成并缓存索引
        # print("46wobuzaihuancun")
        # print(H,W,direction)
        precompute_snake_scan_indices(H, W)
        o, o_inverse, d = index_cache[(H, W, direction)]

    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维
    real_data_reordered = real_data_flattened[:, :, o]  # 根据蛇形扫描索引重新排序

    # 返回重新排列的数据和相关信息
        # 将结果放入列表中返回
    result = []
    result.append(real_data_reordered)
    result.append(o_inverse)
    result.append(d)

    return result


# import torch
#
# # 创建一个简单的 4x4 二维矩阵用于测试
# H, W = 4, 4
# original_data = torch.arange(1, H * W + 1).reshape(1, 1, H, W).float()
# print("Original Data:\n", original_data)
#
# # 调用水平反向蛇形扫描函数
# precompute_snake_scan_indices(H, W)  # 预计算索引
# snake_result = snake_scan_with_cache(original_data, direction="h_flip")
#
# # 提取重排后的数据、反向索引和方向索引
# reordered_data = snake_result[0]
# o_inverse = snake_result[1]
# d = snake_result[2]
#
# # 将重排后的数据打印
# print("Reordered Data (h_flip):\n", reordered_data)
#
# # 使用反向索引将数据还原为原始顺序
# # reconstructed_data = torch.zeros_like(original_data.flatten(2))
# # reconstructed_data[:, :, o_inverse] = reordered_data  # 根据 o_inverse 反向重排
# # 直接使用 o_inverse 作为索引来还原数据顺序
# reconstructed_data = reordered_data[:, :, o_inverse]
# reconstructed_data = reconstructed_data.reshape(1, 1, H, W)  # 重新调整为 H x W
#
# # 打印还原的二维矩阵
# print("Reconstructed Data (h_flip):\n", reconstructed_data)
