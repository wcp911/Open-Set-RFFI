import random

# Step 1: 生成第一个矩阵
matrix_1 = []

for _ in range(5):  # 数字代表蒙特卡洛次数
    row = sorted(random.sample(range(10), 6))  # 第一个数字代表总类别数，第二个数字代表已知类别数
    matrix_1.append(row)

# 打印第一个矩阵
print("第一个矩阵 (5x6):")
for row in matrix_1:
    print(row)

# Step 2: 生成第二个矩阵
matrix_2 = []

for row in matrix_1:
    remaining_numbers = list(set(range(10)) - set(row))
    extra_numbers = sorted(remaining_numbers[:4])  # 挑选剩余数中的前四个
    new_row = row + extra_numbers
    matrix_2.append(new_row)

# 打印第二个矩阵
print("\n第二个矩阵 (5x10):")
for row in matrix_2:
    print(row)
