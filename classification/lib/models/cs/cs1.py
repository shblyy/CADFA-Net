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
        print("woaiyy")
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
    precompute_snake_scan_indices(size, size)
# 使用缓存索引的函数示例
def snake_scan_with_cache(real_data, direction="h"):
    batch_size, channels, H, W = real_data.shape

    if (H, W, direction) in index_cache:
        # print("12wozaihuancun")
        o, o_inverse, d = index_cache[(H, W, direction)]
    else:
        # 如果没有缓存，生成并缓存索引
        print("46wobuzaihuancun")
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


import torch

# 定义一个简单的 4x4 测试矩阵，batch_size=1, channels=1, H=4, W=4
H, W = 4, 4
token_size = (H, W)
test_data = torch.arange(1, H * W + 1).reshape(1, 1, H, W).float()
print("Original Data:\n", test_data)

# 定义扫描和逆向函数类
class SnakeScanTest:
    def __init__(self, token_size):
        self.token_size = token_size

    def scan(self, x, direction='h'):
        H, W = self.token_size
        if direction == 'h':
            xflatten = snake_scan_with_cache(x, 'h')
            return xflatten
        elif direction == 'h_flip':
            xflatten = snake_scan_with_cache(x, 'h_flip')
            return xflatten
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h', xs2=[1,2]):
        H, W = self.token_size
        xs2_tensor = torch.tensor(xs2)

        if direction == 'h' or direction == 'h_flip':
            # x_reversed = x[:, :, xs2_tensor]
            # 根据逆向索引 xs2_tensor 还原数据
            x_reversed = x[:, :, xs2_tensor]
            # x_reversed = torch.zeros_like(x)
            # x_reversed[:, :, xs2_tensor] = x
            return x_reversed
        else:
            raise RuntimeError(f'Direction {direction} not found.')

# 初始化测试类
scan_test = SnakeScanTest(token_size)

# 对测试数据进行扫描（h 和 h_flip 方向）
scan_result_h = scan_test.scan(test_data, direction='h')
scan_result_h_flip = scan_test.scan(test_data, direction='h_flip')

# 打印扫描结果
print("Scan Result (h):\n", scan_result_h[0])
print("Scan Result (h_flip):\n", scan_result_h_flip[0])

# 使用逆向索引还原数据
reverse_result_h = scan_test.reverse(scan_result_h[0], direction='h', xs2=scan_result_h[1])
reverse_result_h_flip = scan_test.reverse(scan_result_h_flip[0], direction='h_flip', xs2=scan_result_h_flip[1])

# 打印还原结果
print("Reversed Data (h):\n", reverse_result_h.reshape(1, 1, H, W))
print("Reversed Data (h_flip):\n", reverse_result_h_flip.reshape(1, 1, H, W))

# 检查还原数据是否与原始数据相同
if torch.equal(test_data, reverse_result_h.reshape(1, 1, H, W)):
    print("Test Passed for h direction: The reconstructed data matches the original data.")
else:
    print("Test Failed for h direction: The reconstructed data does not match the original data.")

if torch.equal(test_data, reverse_result_h_flip.reshape(1, 1, H, W)):
    print("Test Passed for h_flip direction: The reconstructed data matches the original data.")
else:
    print("Test Failed for h_flip direction: The reconstructed data does not match the original data.")
