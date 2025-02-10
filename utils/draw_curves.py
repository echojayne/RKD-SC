# 绘制loss曲线并保存

import matplotlib.pyplot as plt
import numpy as np

def draw_curves(loss, save_path, smooth=False):
    if smooth:
        loss = smooth_curve_n(loss, 3)
    plt.plot(loss, label='loss')
    plt.legend()
    # 纵轴范围
    # plt.ylim(0, 0.1)
    plt.savefig(save_path)
    plt.show()
    plt.close()

#去除奇异值
    
def remove_outliers(data):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    data = data[np.abs(data - mean) <= 3 * std]
    return data

# 平滑曲线
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# n阶平滑曲线
def smooth_curve_n(points, n=3):
    for i in range(n):
        points = smooth_curve(points)
    return points

if __name__ == '__main__':
    # 读入LOSS

    loss = np.loadtxt('/home/ubuntu/users/dky/CLIP-KD/results/curve_data/cm_train_loss.txt')
    draw_curves(loss, '/home/ubuntu/users/dky/CLIP-KD/results/curves/loss_CT.png', False)
