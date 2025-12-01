import torch
#块状蛇形扫描
# 定义蛇形扫描块反转函数
def snake_scan_blocks_reverse(real_data, block_size):
    batch_size, channels, H, W = real_data.shape
    L = H * W

    # 初始化完整的索引和方向
    o_full = []
    d_full = []
    o_full_inverse = [-1 for _ in range(L)]

    # 计算每个块内的蛇形扫描顺序
    def snake_scan_block_indices(n):
        indices = []
        directions = []
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
        return indices, directions

    # 获取块内扫描顺序
    block_indices, block_directions = snake_scan_block_indices(block_size)

    # 按块划分，并逐块填充到完整索引中
    block_row_count = H // block_size
    block_col_count = W // block_size

    for br in reversed(range(block_row_count)):
        print("br", br)
        for bc in reversed(range(block_col_count)):
            print("bc", bc)
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

    # 用 0 代替第一个方向
    d_full = [0] + d_full[:-1]

    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维
    real_data_reordered = real_data_flattened[:, :, o_full]  # 根据蛇形扫描索引重新排序

    # 返回重新排列的数据和相关信息
    result = []
    result.append(real_data_reordered)
    result.append(o_full_inverse)
    result.append(d_full)

    return result


#----------------这是正向的---------------------------------
def snake_scan_blocks(real_data, block_size):
    batch_size, channels, H, W = real_data.shape
    L = H * W

    # 初始化完整的索引和方向
    o_full = []
    d_full = []
    o_full_inverse = [-1 for _ in range(L)]

    # 计算每个块内的蛇形扫描顺序
    def snake_scan_block_indices(n):
        indices = []
        directions = []
        i, j, j_d = 0, 0, "right"
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
    block_indices, block_directions = snake_scan_block_indices(block_size)

    # 按块划分，并逐块填充到完整索引中
    block_row_count = H // block_size
    block_col_count = W // block_size

    for br in range(block_row_count):
        print("br",br)
        for bc in range(block_col_count):
            print("bc", bc)
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

    # 展平并重新排列每个通道的特征图
    real_data_flattened = real_data.flatten(2)  # 将 H 和 W 维度展平为一维
    real_data_reordered = real_data_flattened[:, :, o_full]  # 根据蛇形扫描索引重新排序

    # 返回重新排列的数据和相关信息
    result = []
    result.append(real_data_reordered)
    result.append(o_full_inverse)
    result.append(d_full)

    return result




# 创建一个测试输入张量 (batch_size, channels, H, W)
batch_size, channels, H, W = 1, 1, 8, 8
real_data = torch.arange(batch_size * channels * H * W).view(batch_size, channels, H, W)

# 设置块大小
block_size = 4

# 调用 snake_scan_blocks_reverse 函数
result = snake_scan_blocks_reverse(real_data, block_size)

# 获取结果
real_data_reordered, o_full_inverse, d_full = result

# 原始数据
print("原始数据：")
print(real_data.view(H, W))

# 重排后的数据
print("\n重排后的数据：")
print(real_data_reordered.view(H, W))

# 反向重排后的数据（将重排后的数据重新恢复为原始顺序）
# real_data_reverse = real_data_reordered.new_zeros(real_data_reordered.shape)
# for i, idx in enumerate(o_full_inverse):
#     real_data_reverse[:, :, idx] = real_data_reordered[:, :, i]
# 直接使用 o_full_inverse 作为索引来还原数据顺序
real_data_reverse = real_data_reordered[:, :, o_full_inverse]


print("\n反向重排后的数据：")
print(real_data_reverse.view(H, W))

# 输出方向列表
print("\n方向列表 (d_full)：")
print(d_full)



import torch

# 创建一个测试输入张量 (batch_size, channels, H, W)
batch_size, channels, H, W = 1, 1, 8, 8  # 示例尺寸 8x8
real_data = torch.arange(batch_size * channels * H * W).view(batch_size, channels, H, W)

# 设置块大小
block_size = 4

# 调用 snake_scan_blocks 函数
result = snake_scan_blocks(real_data, block_size)

# 获取函数输出
real_data_reordered, o_full_inverse, d_full = result

# 打印原始二维数据
print("原始数据：")
print(real_data.view(H, W))

# 打印重排后的二维数据
print("\n重排后的数据：")
print(real_data_reordered.view(H, W))

# 打印全局索引的逆序列表
print("\n全局索引的逆序列表 (o_full_inverse)：")
print(o_full_inverse)

# 打印方向列表
print("\n方向列表 (d_full)：")
print(d_full)

# 还原重排后的数据以验证逆向索引
# real_data_restored = torch.zeros_like(real_data_reordered)
# for i, idx in enumerate(o_full_inverse):
#     real_data_restored[:, :, idx] = real_data_reordered[:, :, i]

real_data_restored = real_data_reordered[:, :, o_full_inverse]

print("\n还原后的数据（应与原始数据相同）：")
print(real_data_restored.view(H, W))
