from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from convlstm_model import *
from input_data import *

import os

# 获取当前工作目录
current_directory = os.getcwd()

# 列出当前目录下的所有文件和文件夹
files_and_folders = os.listdir(current_directory)

# 打印文件和文件夹的名称
for item in files_and_folders:
    print(item)

sores = []
criterion = nn.MSELoss()
preds1 = np.zeros((539,12,15))
preds1 = np.expand_dims(preds1, axis=1)
Qnet_test1 = np.zeros((539,1,7,12,15))
mld_test1 = np.zeros((539,1,7,12,15))
u_test1 = np.zeros((539,1,7,12,15))
v_test1 = np.zeros((539,1,7,12,15))
sst1_test1 = np.zeros((539,1,7,12,15))
T_d_test1 = np.zeros((539,1,7,12,15))
u_d_test1 = np.zeros((539,1,7,12,15))

def rmse(y_true, y_preds):
    # 计算平方差
    squared_diffs = (y_preds - y_true) ** 2
    # 计算平方差的均值
    mean_squared_error = torch.mean(squared_diffs)
    # 返回均方根误差
    return torch.sqrt(mean_squared_error)


dt_x_y = 1079
true_day = 3645

checkpoint = torch.load('./convlstm_classical_PINN-based_train_modes.pt')
print(checkpoint.keys())
input_dim = 1

hidden_dim = (16, 16, 16)

kernel_size = (3, 3)
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvLSTM(input_dim, hidden_dim, kernel_size)
model.load_state_dict(checkpoint)
model=model.to(device)
# test

model.eval()
for k in range(1):
    for i, data in tqdm(enumerate(testloader)):
        data,label = data
        data = data.to(device)
        label = label.to(device)
        data1 = data[:,1:3,:,:,:]

        preds2 = model(data1)
        dT_x = torch.autograd.grad(outputs=preds2, inputs=xx_test, grad_outputs=torch.ones_like(preds2), retain_graph=True,create_graph=True)[0]
        dT_y = torch.autograd.grad(outputs=preds2, inputs=yy_test, grad_outputs=torch.ones_like(preds2), retain_graph=True,create_graph=True)[0]
        aerfa = 0.1
        beta = 0.9

        dT_x = dT_x[:dt_x_y]
        dT_y = dT_y[:dt_x_y]

        dT_x1 = dT_x[:, :, 0, :, :]
        dT_x2 = dT_x[:, :, 1, :, :]
        dT_x3 = dT_x[:, :, 2, :, :]
        dT_x4 = dT_x[:, :, 3, :, :]
        dT_x5 = dT_x[:, :, 4, :, :]
        dT_x6 = dT_x[:, :, 5, :, :]
        dT_x7 = dT_x[:, :, 6, :, :]
        dT_y1 = dT_y[:, :, 0, :, :]
        dT_y2 = dT_y[:, :, 1, :, :]
        dT_y3 = dT_y[:, :, 2, :, :]
        dT_y4 = dT_y[:, :, 3, :, :]
        dT_y5 = dT_y[:, :, 4, :, :]
        dT_y6 = dT_y[:, :, 5, :, :]
        dT_y7 = dT_y[:, :, 6, :, :]

        # Construction data
        Qnet_test = data[:, 3, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        mld_test = data[:, 4, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        u_test = data[:, 5, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        v_test = data[:, 6, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        u_d_test = data[:, 8, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        sst1_test = data[:, 0, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()
        T_d_test = data[:, 7, :, :, :].reshape(-1, 1, 7, 6, 27).cpu()

        Qnet_test2 = torch.Tensor(Qnet_test)
        mld_test2 = torch.Tensor(mld_test)
        u_test2 = torch.Tensor(u_test)
        v_test2 = torch.Tensor(v_test)
        u_d_test2 = torch.Tensor(u_d_test)
        sst1_test2 = torch.Tensor(sst1_test)
        T_d_test2 = torch.Tensor(T_d_test)

    aerfa = 0.1
    beta = 0.9
    pred1 = ((86400 * (Qnet_test2[:, :, 0, :, :] / (1025 * 4000 * mld_test2[:, :, 0, :, :])) - (u_test2[:, :, 0, :, :] * dT_x1) - (v_test2[:, :, 0, :, :] * dT_y1) - (u_d_test2[:, :, 0, :, :] * (sst1_test2[:, :, 0, :, :] - T_d_test2[:, :, 0, :, :]) / mld_test2[:, :, 0, :, :])) + sst1_test2[:, :, 0, :, :]).cuda()
    pred2 = ((86400 * (Qnet_test2[:, :, 1, :, :] / (1025 * 4000 * mld_test2[:, :, 1, :, :])) - (u_test2[:, :, 1, :, :] * dT_x2) - (v_test2[:, :, 1, :, :] * dT_y2) - (u_d_test2[:, :, 1, :, :] * (sst1_test2[:, :, 1, :, :] - T_d_test2[:, :, 1, :, :]) / mld_test2[:, :, 1, :, :])) + sst1_test2[:, :, 1, :, :]).cuda()
    pred3 = ((86400 * (Qnet_test2[:, :, 2, :, :] / (1025 * 4000 * mld_test2[:, :, 2, :, :])) - (u_test2[:, :, 2, :, :] * dT_x3) - (v_test2[:, :, 2, :, :] * dT_y3) - (u_d_test2[:, :, 2, :, :] * (sst1_test2[:, :, 2, :, :] - T_d_test2[:, :, 2, :, :]) / mld_test2[:, :, 2, :, :])) + sst1_test2[:, :, 2, :, :]).cuda()
    pred4 = ((86400 * (Qnet_test2[:, :, 3, :, :] / (1025 * 4000 * mld_test2[:, :, 3, :, :])) - (u_test2[:, :, 3, :, :] * dT_x4) - (v_test2[:, :, 3, :, :] * dT_y4) - (u_d_test2[:, :, 3, :, :] * (sst1_test2[:, :, 3, :, :] - T_d_test2[:, :, 3, :, :]) / mld_test2[:, :, 3, :, :])) + sst1_test2[:, :, 3, :, :]).cuda()
    pred5 = ((86400 * (Qnet_test2[:, :, 4, :, :] / (1025 * 4000 * mld_test2[:, :, 4, :, :])) - (u_test2[:, :, 4, :, :] * dT_x5) - (v_test2[:, :, 4, :, :] * dT_y5) - (u_d_test2[:, :, 4, :, :] * (sst1_test2[:, :, 4, :, :] - T_d_test2[:, :, 4, :, :]) / mld_test2[:, :, 4, :, :])) + sst1_test2[:, :, 4, :, :]).cuda()
    pred6 = ((86400 * (Qnet_test2[:, :, 5, :, :] / (1025 * 4000 * mld_test2[:, :, 5, :, :])) - (u_test2[:, :, 5, :, :] * dT_x6) - (v_test2[:, :, 5, :, :] * dT_y6) - (u_d_test2[:, :, 5, :, :] * (sst1_test2[:, :, 5, :, :] - T_d_test2[:, :, 5, :, :]) / mld_test2[:, :, 5, :, :])) + sst1_test2[:, :, 5, :, :]).cuda()
    pred7 = ((86400 * (Qnet_test2[:, :, 6, :, :] / (1025 * 4000 * mld_test2[:, :, 6, :, :])) - (u_test2[:, :, 6, :, :] * dT_x7) - (v_test2[:, :, 6, :, :] * dT_y7) - (u_d_test2[:, :, 6, :, :] * (sst1_test2[:, :, 6, :, :] - T_d_test2[:, :, 6, :, :]) / mld_test2[:, :, 6, :, :])) + sst1_test2[:, :, 6, :, :]).cuda()
    pred8 = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * pred1 + beta * pred2) + beta * pred3) + beta * pred4) + beta * pred5) + beta * pred6) + beta * pred7

# 首先确保pred_test1和test_label已经在适当的设备上
pred_test1 = pred8.reshape(-1,1).to(device)
test_label = test_label11.reshape(-1,1).to(device)


# 计算RMSE
s = rmse(test_label, pred_test1)  # 现在 s 是一个 Tensor，且保留在 GPU 上（如果设备是 CUDA 的话）
print('RMSE: {:.3f}'.format(s.item()))  # 使用 .item() 将单个 Tensor 值转换为 Python 标量进行打印


# 使用PyTorch进行MAE, MSE, MAPE的计算，确保所有操作都在Tensor上执行
MAE = torch.mean(torch.abs(test_label - pred_test1).cpu())
print('MAE: {:.3f}'.format(MAE.item()))

MSE = torch.mean(((test_label - pred_test1) ** 2).cpu())
print('MSE: {:.3f}'.format(MSE.item()))

MAPE = torch.mean(torch.abs((test_label - pred_test1) / test_label).cpu())
print('MAPE: {:.3f}'.format(MAPE.item()))

# 如果需要使用NumPy进行计算，确保先转移到CPU
test_label_np = test_label.cpu().numpy()  # 假设 test_label 不需要梯度，所以可以直接转换

# 对 pred_test1 使用 .detach() 方法后再转移到 CPU 并转换为 NumPy 数组
pred_test1_np = pred_test1.detach().cpu().numpy()

# 使用NumPy进行相关性计算
x_hat = np.mean(test_label_np)
y_hat = np.mean(pred_test1_np)

fenzi = np.sum((test_label_np - x_hat) * (pred_test1_np - y_hat))
fenmu = np.sqrt((np.sum((test_label_np - x_hat)**2) * np.sum((pred_test1_np - y_hat)**2)))

cor = fenzi / fenmu
print('Correlation: {:.3f}'.format(cor))

ture_1day = sst[valid_size + 0: true_day - 6,:,:,:]
ture_2day = sst[valid_size + 1: true_day - 5,:,:,:]
ture_3day = sst[valid_size + 2: true_day - 4,:,:,:]
ture_4day = sst[valid_size + 3: true_day - 3,:,:,:]
ture_5day = sst[valid_size + 4: true_day - 2,:,:,:]
ture_6day = sst[valid_size + 5: true_day - 1,:,:,:]
ture_7day = sst[valid_size + 6: true_day,:,:,:]
true_day = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * ture_1day + beta * ture_2day) + beta * ture_3day) + beta * ture_4day) + beta * ture_5day) + beta * ture_6day) + beta * ture_7day
ture_day = torch.tensor(true_day).cuda()

Tend_true = pred8 - ture_day

Qnet = aerfa*(aerfa*(aerfa*(aerfa*(aerfa*(aerfa*(86400*(Qnet_test2[:, :, 0, :, :] / (1025 * 4000 * mld_test2[:, :, 0, :, :]))) + beta*(86400*(Qnet_test2[:, :, 1, :, :] / (1025 * 4000 * mld_test2[:, :, 1, :, :])))) + beta*(86400*(Qnet_test2[:, :, 2, :, :] / (1025 * 4000 * mld_test2[:, :, 2, :, :])))
             ) + beta*(86400*(Qnet_test2[:, :, 3, :, :] / (1025 * 4000 * mld_test2[:, :, 3, :, :])))) + beta*((86400*(Qnet_test2[:, :, 4, :, :] / (1025 * 4000 * mld_test2[:, :, 4, :, :]))))) + beta*(86400*(Qnet_test2[:, :, 5, :, :] / (1025 * 4000 * mld_test2[:, :, 5, :, :])))) +beta*(86400*(Qnet_test2[:, :, 6, :, :] / (1025 * 4000 * mld_test2[:, :, 6, :, :])))

ZAdv = aerfa * (aerfa * (aerfa*(aerfa*(aerfa*(aerfa*((u_test2[:, :, 0, :, :] * dT_x1)) + beta*(u_test2[:, :, 1, :, :] * dT_x2)) + beta*(u_test2[:, :, 2, :, :] * dT_x3)) + beta*(u_test2[:, :, 3, :, :] * dT_x4))+ beta *(u_test2[:, :, 4, :, :] * dT_x5)
                ) + beta*(u_test2[:, :, 5, :, :] * dT_x6)) + beta*((u_test2[:, :, 6, :, :] * dT_x7))

MAdv = aerfa * (aerfa * (aerfa*(aerfa*(aerfa*(aerfa*((v_test2[:, :, 0, :, :] * dT_y1)) + beta*(v_test2[:, :, 1, :, :] * dT_y2)) + beta*(v_test2[:, :, 2, :, :] * dT_y3)) + beta*(v_test2[:, :, 3, :, :] * dT_y4))+ beta *(v_test2[:, :, 4, :, :] * dT_y5)
                ) + beta*(v_test2[:, :, 5, :, :] * dT_y6)) + beta*((v_test2[:, :, 6, :, :] * dT_y7))

VAdv = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * ((u_d_test2[:, :, 0, :, :] * (sst1_test2[:, :, 0, :, :] - T_d_test2[:, :, 0, :, :]) / mld_test2[:, :, 0, :, :])) + beta*(u_d_test2[:, :, 1, :, :] * (sst1_test2[:, :, 1, :, :] - T_d_test2[:, :, 1, :, :]) / mld_test2[:, :, 1, :, :])
                ) + beta * (u_d_test2[:, :, 2, :, :] * (sst1_test2[:, :, 2, :, :] - T_d_test2[:, :, 2, :, :]) / mld_test2[:, :, 2, :, :])) + beta * (u_d_test2[:, :, 3, :, :] * (sst1_test2[:, :, 3, :, :] - T_d_test2[:, :, 3, :, :]) / mld_test2[:, :, 3, :, :])
                ) + beta * (u_d_test2[:, :, 4, :, :] * (sst1_test2[:, :, 4, :, :] - T_d_test2[:, :, 4, :, :]) / mld_test2[:, :, 4, :, :])) + beta * (u_d_test2[:, :, 5, :, :] * (sst1_test2[:, :, 5, :, :] - T_d_test2[:, :, 5, :, :]) / mld_test2[:, :, 5, :, :])) + beta * (u_d_test2[:, :, 6, :, :] * (sst1_test2[:, :, 6, :, :] - T_d_test2[:, :, 6, :, :]) / mld_test2[:, :, 6, :, :])

Tend = Qnet - ZAdv - MAdv - VAdv

Qnet_1 = Qnet.detach().cpu().numpy()
ZAdv_1 = ZAdv.detach().cpu().numpy()
MAdv_1 = MAdv.detach().cpu().numpy()
VAdv_1 = VAdv.detach().cpu().numpy()
Tend1 = Tend.detach().cpu().numpy()


Tend_true_1 = Tend_true.detach().cpu().numpy()
print('Tend_no_R:{}'.format(np.mean(Tend1)))
print('Qnet:{}'.format(np.mean(Qnet_1)))
print('ZAdv:{}'.format(np.mean(ZAdv_1)))
print('MAdv:{}'.format(np.mean(MAdv_1)))
print('VAdv:{}'.format(np.mean(VAdv_1)))
print('Tend_true:{}'.format(np.mean(Tend_true_1)))