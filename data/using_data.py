import numpy as np
import os

# 设置文件夹路径
#directory = r'E:\硕\PINN实现\STPDE_NET-main\STPDE_NET-main\data'
d = r'E:\硕\PINN实现\STPDE_NET-main\STPDE_NET-main\code\cnn'

# 获取文件夹下所有的文件和文件夹的名称
entries = os.listdir(d)

# 过滤出文件，排除文件夹
files = [entry for entry in entries if os.path.isfile(os.path.join(d, entry))]

# 打印文件名
for file in files:
    print(file)

# 加载 `.npz` 文件
data = np.load('{}'.format("label.npz"),allow_pickle=True)
print(data.files)
print(data["sst"])