import torch

# 初始化缓存字典
index_cache = {}

def precompute_snake_block_indices(H, W, block_size, flip=False):
    """
    生成并缓存块状蛇形扫描的索引。
    输入:
        H (int): 图像高度
        W (int): 图像宽度
        block_size (int): 扫描块的大小
        flip (bool): 是否执行反向扫描 (flip=True 为反向)
    """
    cache_key = (H, W, block_size, flip)

    # 检查是否已经缓存
    if cache_key in index_cache:
        return  # 已缓存则直接返回

    # 初始化完整的索引和方向
    o_full = []
    d_full = []
    o_full_inverse = [-1 for _ in range(H * W)]

    # 定义块内蛇形扫描顺序生成函数
    def snake_scan_block_indices(n, reverse=False):
        indices = []
        directions = []
        if reverse:
            i, j, j_d = n - 1, n - 1, "left"  # 从右下角开始
            while i >= 0:
                idx = i * n + j
                indices.append(idx)
                if j_d == "left":
                    if j > 0:
                        j -= 1
                        directions.append(2)  # 左
                    else:
                        i -= 1
                        directions.append(3)  # 上
                        j_d = "right"
                else:
                    if j < n - 1:
                        j += 1
                        directions.append(1)  # 右
                    else:
                        i -= 1
                        directions.append(3)  # 上
                        j_d = "left"
            directions = [0] + directions[:-1]  # 第一个方向设为 0
        else:
            i, j, j_d = 0, 0, "right"  # 从左上角开始
            while i < n:
                idx = i * n + j
                indices.append(idx)
                if j_d == "right":
                    if j < n - 1:
                        j += 1
                        directions.append(1)  # 右
                    else:
                        i += 1
                        directions.append(4)  # 下
                        j_d = "left"
                else:
                    if j > 0:
                        j -= 1
                        directions.append(2)  # 左
                    else:
                        i += 1
                        directions.append(4)  # 下
                        j_d = "right"
            directions = [0] + directions[:-1]  # 第一个方向设为 0
        return indices, directions

    # 获取块内扫描顺序
    block_indices, block_directions = snake_scan_block_indices(block_size, reverse=flip)

    # 按块划分，并逐块填充到完整索引中
    block_row_count = H // block_size
    block_col_count = W // block_size

    if flip:
        for br in reversed(range(block_row_count)):
            for bc in reversed(range(block_col_count)):
                for idx, direction in zip(block_indices, block_directions):
                    # 计算全局索引
                    i = (idx // block_size) + br * block_size
                    j = (idx % block_size) + bc * block_size
                    global_idx = i * W + j

                    # 添加到完整扫描顺序中
                    o_full_inverse[global_idx] = len(o_full)
                    # o_full_inverse[len(o_full)] = global_idx  # 记录逆向索引
                    o_full.append(global_idx)
                    d_full.append(direction)
    else:
        for br in range(block_row_count):
            for bc in range(block_col_count):
                for idx, direction in zip(block_indices, block_directions):
                    # 计算全局索引
                    i = (idx // block_size) + br * block_size
                    j = (idx % block_size) + bc * block_size
                    global_idx = i * W + j

                    # 添加到完整扫描顺序中
                    o_full_inverse[global_idx] = len(o_full)
                    o_full.append(global_idx)
                    d_full.append(direction)

    # 用 0 代替第一个方向
    d_full = [0] + d_full[:-1]

    # 将生成的索引缓存起来
    index_cache[cache_key] = (o_full, o_full_inverse, d_full)

def snake_scan_blocks_with_cache(real_data, block_size, flip=False):
    """
    使用缓存的块状蛇形扫描函数执行扫描。
    输入:
        real_data (torch.Tensor): 输入张量，形状为 [batch_size, channels, H, W]
        block_size (int): 扫描块的大小
        flip (bool): 是否执行反向扫描 (flip=True 为反向)
    返回:
        result (list): 包含重新排序的数据，逆向索引和方向信息
    """
    batch_size, channels, H, W = real_data.shape
    # 检查 H 和 W 是否为 tensor，如果是则转换为整数
    if isinstance(H, torch.Tensor):
        H = H.item()
    if isinstance(W, torch.Tensor):
        W = W.item()
    cache_key = (H, W, block_size, flip)

    # 确保索引已缓存
    if cache_key not in index_cache:
        precompute_snake_block_indices(H, W, block_size, flip=flip)

    # 从缓存中获取索引
    o_full, o_full_inverse, d_full = index_cache[cache_key]
    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维
    real_data_reordered = real_data_flattened[:, :, o_full]  # 根据蛇形扫描索引重新排序

    # 返回重新排列的数据和相关信息
    result = [real_data_reordered, o_full_inverse, d_full]
    return result

# 预计算常用尺寸和块大小的索引
# for size in [56, 28, 14, 7]:
#     for block_size in [2, 7]:  # 预定义一些常用的块大小
#         for flip in [False, True]:  # 正向和反向扫描
#             precompute_snake_block_indices(size, size, block_size, flip=flip)
#
# # 示例输入张量，假设尺寸为 8x8
# real_data = torch.arange(1, 65).reshape(1, 1, 8, 8).float()
# block_size = 4
# flip = False
#
# # 调用块状蛇形扫描函数
# result = snake_scan_blocks_with_cache(real_data, block_size, flip=flip)
# real_data_reordered, o_full_inverse, d_full = result
#
# # 打印扫描前、扫描后和反转还原后的二维数据
# print("Original Data:\n", real_data.squeeze())  # 打印扫描前的二维数据
#
# # 打印扫描后的二维数据
# print("Reordered Data:\n", real_data_reordered.reshape(1, 8, 8).squeeze())
#
# # 根据 o_full_inverse 反转回原始顺序
# # real_data_reconstructed = torch.zeros_like(real_data_reordered)
# real_data_reconstructed = real_data_reordered[:, :, o_full_inverse]
# print("Reconstructed Data:\n", real_data_reconstructed.reshape(1, 8, 8).squeeze())
#
#
# import torch
#
# # 示例输入张量，假设尺寸为 8x8
# real_data = torch.arange(1, 65).reshape(1, 1, 8, 8).float()  # 8x8 的示例张量
# block_size = 4  # 选择扫描块大小为 4x4
# flip = True  # 使用反向扫描
#
# # 调用块状蛇形扫描函数
# result = snake_scan_blocks_with_cache(real_data, block_size, flip=flip)
# real_data_reordered, o_full_inverse, d_full = result
# # 打印扫描前、扫描后和反转还原后的二维数据
# print("Original Data:\n", real_data.squeeze())  # 打印扫描前的二维数据
#
# # 检查输出形状
# print("Shape of reordered data:", real_data_reordered)
#
# # 如果形状正确为 [1, 1, 64]，则继续，否则调整块大小或输入
# if real_data_reordered.shape[-1] == 64:
#     # 打印扫描前、扫描后的二维数据，以及反转还原后的二维数据
#     print("Original Data:\n", real_data.squeeze())  # 扫描前的二维数据
#
#     # 打印扫描后的二维数据
#     print("Reordered Data (after reverse scan):\n", real_data_reordered.reshape(1, 8, 8).squeeze())
#
#     real_data_reconstructed = real_data_reordered[:, :, o_full_inverse]
#     print("Reconstructed Data:\n", real_data_reconstructed.reshape(1, 8, 8).squeeze())
# else:
#     print("Reordered data shape does not match expected 8x8 size; please adjust input or block size.")
