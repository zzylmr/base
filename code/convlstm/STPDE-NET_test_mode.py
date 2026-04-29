from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from convlstm_model import *
from input_data import *

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sores = []
criterion = nn.MSELoss()
preds1 = np.zeros((538,12,15))
preds1 = np.expand_dims(preds1, axis=1)
Qnet_test1 = np.zeros((538,1,7,12,15))
mld_test1 = np.zeros((538,1,7,12,15))
u_test1 = np.zeros((538,1,7,12,15))
v_test1 = np.zeros((538,1,7,12,15))
sst1_test1 = np.zeros((538,1,7,12,15))
T_d_test1 = np.zeros((538,1,7,12,15))
u_d_test1 = np.zeros((538,1,7,12,15))
def rmse(y_true, y_preds):
    # Ensure the tensors are detached and converted to NumPy arrays
    y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else y_true
    y_preds = y_preds.detach().cpu().numpy() if torch.is_tensor(y_preds) else y_preds
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_preds))


checkpoint = torch.load('./convlstm_STPDE-NET_train_mode.pth')

input_dim = 1

hidden_dim = (16,16,16)

kernel_size = (3, 3)

model1 = ConvLSTM(input_dim, hidden_dim,  kernel_size).to(device)

model1.load_state_dict(checkpoint['state_dict'])

#test

model1.eval()
for k in range(1):
    for i, data in tqdm(enumerate(testloader)):
        data,label = data
        data = data
        label = label
        data1 = data[:,1:3,:,:,:].to(device)
        print(f"Shape of data1: {data1.shape}")
        preds2 = model1(data1)
        dT_x = torch.autograd.grad(outputs=preds2, inputs=xx_test, grad_outputs=torch.ones_like(preds2), retain_graph=True,create_graph=True)[0]
        dT_y = torch.autograd.grad(outputs=preds2, inputs=yy_test, grad_outputs=torch.ones_like(preds2), retain_graph=True,create_graph=True)[0]

        dT_x = dT_x[:1079]
        dT_y = dT_y[:1079]

        aerfa = 0.1
        beta = 0.9

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

        pred1 = ((86400 *(Qnet_test2[:, :, 0, :, :] / (1025 * 4000 * mld_test2[:, :, 0, :, :])) - (u_test2[:, :, 0, :, :] * dT_x1) - (v_test2[:, :, 0, :, :] * dT_y1) - (u_d_test2[:, :, 0, :, :] * (sst1_test2[:, :, 0, :, :] - T_d_test2[:, :, 0, :, :]) / mld_test2[:, :, 0, :, :])) + sst1_test2[:, :, 0, :, :])
        pred2 = ((86400 *(Qnet_test2[:, :, 1, :, :] / (1025 * 4000 * mld_test2[:, :, 1, :, :])) - (u_test2[:, :, 1, :, :] * dT_x2) - (v_test2[:, :, 1, :, :] * dT_y2) - (u_d_test2[:, :, 1, :, :] * (sst1_test2[:, :, 1, :, :] - T_d_test2[:, :, 1, :, :]) / mld_test2[:, :, 1, :, :])) + sst1_test2[:, :, 1, :, :])
        pred3 = ((86400 *(Qnet_test2[:, :, 2, :, :] / (1025 * 4000 * mld_test2[:, :, 2, :, :])) - (u_test2[:, :, 2, :, :] * dT_x3) - (v_test2[:, :, 2, :, :] * dT_y3) - (u_d_test2[:, :, 2, :, :] * (sst1_test2[:, :, 2, :, :] - T_d_test2[:, :, 2, :, :]) / mld_test2[:, :, 2, :, :])) + sst1_test2[:, :, 2, :, :])
        pred4 = ((86400 *(Qnet_test2[:, :, 3, :, :] / (1025 * 4000 * mld_test2[:, :, 3, :, :])) - (u_test2[:, :, 3, :, :] * dT_x4) - (v_test2[:, :, 3, :, :] * dT_y4) - (u_d_test2[:, :, 3, :, :] * (sst1_test2[:, :, 3, :, :] - T_d_test2[:, :, 3, :, :]) / mld_test2[:, :, 3, :, :])) + sst1_test2[:, :, 3, :, :])
        pred5 = ((86400 *(Qnet_test2[:, :, 4, :, :] / (1025 * 4000 * mld_test2[:, :, 4, :, :])) - (u_test2[:, :, 4, :, :] * dT_x5) - (v_test2[:, :, 4, :, :] * dT_y5) - (u_d_test2[:, :, 4, :, :] * (sst1_test2[:, :, 4, :, :] - T_d_test2[:, :, 4, :, :]) / mld_test2[:, :, 4, :, :])) + sst1_test2[:, :, 4, :, :])
        pred6 = ((86400 *(Qnet_test2[:, :, 5, :, :] / (1025 * 4000 * mld_test2[:, :, 5, :, :])) - (u_test2[:, :, 5, :, :] * dT_x6) - (v_test2[:, :, 5, :, :] * dT_y6) - (u_d_test2[:, :, 5, :, :] * (sst1_test2[:, :, 5, :, :] - T_d_test2[:, :, 5, :, :]) / mld_test2[:, :, 5, :, :])) + sst1_test2[:, :, 5, :, :])
        pred7 = ((86400 *(Qnet_test2[:, :, 6, :, :] / (1025 * 4000 * mld_test2[:, :, 6, :, :])) - (u_test2[:, :, 6, :, :] * dT_x7) - (v_test2[:, :, 6, :, :] * dT_y7) - (u_d_test2[:, :, 6, :, :] * (sst1_test2[:, :, 6, :, :] - T_d_test2[:, :, 6, :, :]) / mld_test2[:, :, 6, :, :])) + sst1_test2[:, :, 6, :, :])
        pred8 = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * pred1 + beta * pred2) + beta * pred3) + beta * pred4) + beta * pred5) + beta * pred6) + beta * pred7



        losses = 0
        loss = criterion(pred8, test_label11)
        losses += loss

pred_test1 = pred8.reshape(-1,1)
test_label = test_label11.reshape(-1,1)
s = rmse(test_label, pred_test1)
print('RMSE: {:.3f}'.format(s))

# Convert tensors to NumPy arrays
test_label_np = test_label.detach().cpu().numpy()
pred_test1_np = pred_test1.detach().cpu().numpy()

# Calculate MAE
MAE = np.mean(np.abs(test_label_np - pred_test1_np))
print('MAE: {:.3f}'.format(MAE))

# Calculate MSE
MSE = np.mean(np.square(test_label_np - pred_test1_np))
print('MSE: {:.3f}'.format(MSE))

# Calculate MAPE
MAPE = np.mean(np.abs((test_label_np - pred_test1_np) / test_label_np))
print('MAPE: {:.3f}'.format(MAPE))

# Calculate correlation
x_hat = np.mean(test_label_np)
x = test_label_np
y_hat = np.mean(pred_test1_np)
y = pred_test1_np

fenzi = np.sum((x - x_hat) * (y - y_hat))
fenmu = np.sqrt((np.sum(np.square(x - x_hat))) * (np.sum(np.square(y - y_hat))))

cor = fenzi / fenmu
print('Correlation: {:.3f}'.format(cor))

ture_1day = sst[valid_size + 0: 3639,:,:,:]
ture_2day = sst[valid_size + 1: 3640,:,:,:]
ture_3day = sst[valid_size + 2: 3641,:,:,:]
ture_4day = sst[valid_size + 3: 3642,:,:,:]
ture_5day = sst[valid_size + 4: 3643,:,:,:]
ture_6day = sst[valid_size + 5: 3644,:,:,:]
ture_7day = sst[valid_size + 6: 3645,:,:,:]
true_day = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * ture_1day + beta * ture_2day) + beta * ture_3day) + beta * ture_4day) + beta * ture_5day) + beta * ture_6day) + beta * ture_7day
ture_day = true_day

Tend_true = pred8.detach() - ture_day

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

# Convert tensors to NumPy arrays for final calculations
Qnet_np = Qnet.detach().cpu().numpy()
ZAdv_np = ZAdv.detach().cpu().numpy()
MAdv_np = MAdv.detach().cpu().numpy()
VAdv_np = VAdv.detach().cpu().numpy()
Tend_np = Tend.detach().cpu().numpy()
Tend_true_np = Tend_true.detach().cpu().numpy()


# Print mean values
print('Tend_no_R: {:.3f}'.format(np.mean(Tend_np)))
print('Qnet: {:.3f}'.format(np.mean(Qnet_np)))
print('ZAdv: {:.3f}'.format(np.mean(ZAdv_np)))
print('MAdv: {:.3f}'.format(np.mean(MAdv_np)))
print('VAdv: {:.3f}'.format(np.mean(VAdv_np)))
print('Tend_true: {:.3f}'.format(np.mean(Tend_true_np)))