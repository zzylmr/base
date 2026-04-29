import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from cnn_model import ConvNet
from input_data import testloader, xx_test, yy_test, test_label11, sst, valid_size

# Define the loss criterion
criterion = nn.MSELoss()


# Function to calculate RMSE
def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred=y_preds, y_true=y_true))


# Load the model checkpoint
checkpoint = torch.load('./cnn_STPDE-NET_train_mode.pth')

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model
model = ConvNet(input_dim=3, hidden_dim=1, kernel_size1=(3, 3, 3), padding1=(1, 1, 1), kernel_size2=(1, 3, 3),
                padding2=(0, 1, 1)).to(device)
model.load_state_dict(checkpoint['state_dict'])

# Move the model to the appropriate device
model.to(device)

# Evaluation mode
model.eval()

# Iterate over the test data
for k in range(1):
    test_losses = []
    for i, data in tqdm(enumerate(testloader)):
        data, label = data
        data, label = data.to(device), label.to(device)

        # Process the data
        data1 = data[:, 0:3, :, :, :]
        preds2 = model(data1)

        # Calculate gradients for test data if model training involved gradients
        dT = torch.autograd.grad(outputs=preds2, inputs=data1, grad_outputs=torch.ones_like(preds2), retain_graph=True,
                                 create_graph=True, allow_unused=True)[0]

        if dT is not None:
            dT_x, dT_y = dT[:, 1, :, :, :].reshape(-1, 1, 7, 6, 27), dT[:, 2, :, :, :].reshape(-1, 1, 7, 6, 27)

        dT_x, dT_y = dT_x[:1079], dT_y[:1079]

        aerfa = 0.1
        beta = 0.9

        # Process further tensors
        dT_x1, dT_x2, dT_x3, dT_x4, dT_x5, dT_x6, dT_x7 = [dT_x[:, :, i, :, :] for i in range(7)]
        dT_y1, dT_y2, dT_y3, dT_y4, dT_y5, dT_y6, dT_y7 = [dT_y[:, :, i, :, :] for i in range(7)]

        # Construction data, move to the appropriate device
        Qnet_test = data[:, 3, :, :, :].reshape(-1, 1, 7, 6, 27).to(device)
        mld_test = data[:, 4, :, :, :].reshape(-1, 1, 7, 6, 27).to(device)
        u_test = data[:, 5, :, :, :].reshape(-1, 1, 7, 6, 27).to(device)
        v_test = data[:, 6, :, :, :].reshape(-1, 1, 7, 6, 27).to(device)
        u_d_test = data[:, 8, :, :, :].reshape(-1, 1, 7, 6, 27).to(device)
        sst1_test = data[:, 0, :, :, :].reshape(-1, 1, 7, 6, 27).to(device)
        T_d_test = data[:, 7, :, :, :].reshape(-1, 1, 7, 6, 27).to(device)

        # Calculate predictions
        pred1 = ((86400 * (Qnet_test[:, :, 0, :, :] / (1025 * 4000 * mld_test[:, :, 0, :, :])) - (
                    u_test[:, :, 0, :, :] * dT_x1) - (v_test[:, :, 0, :, :] * dT_y1) - (
                              u_d_test[:, :, 0, :, :] * (sst1_test[:, :, 0, :, :] - T_d_test[:, :, 0, :, :]) / mld_test[
                                                                                                               :, :, 0,
                                                                                                               :,
                                                                                                               :])) + sst1_test[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      0,
                                                                                                                      :,
                                                                                                                      :])
        pred2 = ((86400 * (Qnet_test[:, :, 1, :, :] / (1025 * 4000 * mld_test[:, :, 1, :, :])) - (
                    u_test[:, :, 1, :, :] * dT_x2) - (v_test[:, :, 1, :, :] * dT_y2) - (
                              u_d_test[:, :, 1, :, :] * (sst1_test[:, :, 1, :, :] - T_d_test[:, :, 1, :, :]) / mld_test[
                                                                                                               :, :, 1,
                                                                                                               :,
                                                                                                               :])) + sst1_test[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      1,
                                                                                                                      :,
                                                                                                                      :])
        pred3 = ((86400 * (Qnet_test[:, :, 2, :, :] / (1025 * 4000 * mld_test[:, :, 2, :, :])) - (
                    u_test[:, :, 2, :, :] * dT_x3) - (v_test[:, :, 2, :, :] * dT_y3) - (
                              u_d_test[:, :, 2, :, :] * (sst1_test[:, :, 2, :, :] - T_d_test[:, :, 2, :, :]) / mld_test[
                                                                                                               :, :, 2,
                                                                                                               :,
                                                                                                               :])) + sst1_test[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      2,
                                                                                                                      :,
                                                                                                                      :])
        pred4 = ((86400 * (Qnet_test[:, :, 3, :, :] / (1025 * 4000 * mld_test[:, :, 3, :, :])) - (
                    u_test[:, :, 3, :, :] * dT_x4) - (v_test[:, :, 3, :, :] * dT_y4) - (
                              u_d_test[:, :, 3, :, :] * (sst1_test[:, :, 3, :, :] - T_d_test[:, :, 3, :, :]) / mld_test[
                                                                                                               :, :, 3,
                                                                                                               :,
                                                                                                               :])) + sst1_test[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      3,
                                                                                                                      :,
                                                                                                                      :])
        pred5 = ((86400 * (Qnet_test[:, :, 4, :, :] / (1025 * 4000 * mld_test[:, :, 4, :, :])) - (
                    u_test[:, :, 4, :, :] * dT_x5) - (v_test[:, :, 4, :, :] * dT_y5) - (
                              u_d_test[:, :, 4, :, :] * (sst1_test[:, :, 4, :, :] - T_d_test[:, :, 4, :, :]) / mld_test[
                                                                                                               :, :, 4,
                                                                                                               :,
                                                                                                               :])) + sst1_test[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      4,
                                                                                                                      :,
                                                                                                                      :])
        pred6 = ((86400 * (Qnet_test[:, :, 5, :, :] / (1025 * 4000 * mld_test[:, :, 5, :, :])) - (
                    u_test[:, :, 5, :, :] * dT_x6) - (v_test[:, :, 5, :, :] * dT_y6) - (
                              u_d_test[:, :, 5, :, :] * (sst1_test[:, :, 5, :, :] - T_d_test[:, :, 5, :, :]) / mld_test[
                                                                                                               :, :, 5,
                                                                                                               :,
                                                                                                               :])) + sst1_test[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      5,
                                                                                                                      :,
                                                                                                                      :])
        pred7 = ((86400 * (Qnet_test[:, :, 6, :, :] / (1025 * 4000 * mld_test[:, :, 6, :, :])) - (
                    u_test[:, :, 6, :, :] * dT_x7) - (v_test[:, :, 6, :, :] * dT_y7) - (
                              u_d_test[:, :, 6, :, :] * (sst1_test[:, :, 6, :, :] - T_d_test[:, :, 6, :, :]) / mld_test[
                                                                                                               :, :, 6,
                                                                                                               :,
                                                                                                               :])) + sst1_test[
                                                                                                                      :,
                                                                                                                      :,
                                                                                                                      6,
                                                                                                                      :,
                                                                                                                      :])
        pred8 = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (
                    aerfa * pred1 + beta * pred2) + beta * pred3) + beta * pred4) + beta * pred5) + beta * pred6) + beta * pred7

        # Calculate the loss
        loss = criterion(pred8, test_label11.to(device))
        print(f'Loss: {loss.item()}')

# Calculate additional metrics
pred_test1 = pred8.reshape(-1, 1).cpu().detach().numpy()
test_label = test_label11.reshape(-1, 1).cpu().detach().numpy()
s = rmse(test_label, pred_test1)
print('RMSE: {:.3f}'.format(s))

MAE = np.mean(np.abs(test_label - pred_test1))
print('MAE: {:.3f}'.format(MAE))

MSE = np.mean(np.square(test_label - pred_test1))
print('MSE: {:.3f}'.format(MSE))

MAPE = np.mean(np.abs((test_label - pred_test1) / test_label))
print('MAPE: {:.3f}'.format(MAPE))

x_hat = np.mean(test_label)
y_hat = np.mean(pred_test1)

cor = np.sum((test_label - x_hat) * (pred_test1 - y_hat)) / np.sqrt(
    np.sum(np.square(test_label - x_hat)) * np.sum(np.square(pred_test1 - y_hat)))
print('Correlation: {:.3f}'.format(cor))

ture_1day = sst[valid_size + 0: 3639,:,:,:]
ture_2day = sst[valid_size + 1: 3640,:,:,:]
ture_3day = sst[valid_size + 2: 3641,:,:,:]
ture_4day = sst[valid_size + 3: 3642,:,:,:]
ture_5day = sst[valid_size + 4: 3643,:,:,:]
ture_6day = sst[valid_size + 5: 3644,:,:,:]
ture_7day = sst[valid_size + 6: 3645,:,:,:]

# Calculate Tend_true and Tend
true_day = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (
            aerfa * ture_1day + beta * ture_2day) + beta * ture_3day) + beta * ture_4day) + beta * ture_5day) + beta * ture_6day) + beta * ture_7day
Tend_true = pred8 - torch.tensor(true_day).to(device)

Qnet = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (
            aerfa * (86400 * (Qnet_test[:, :, 0, :, :] / (1025 * 4000 * mld_test[:, :, 0, :, :]))) + beta * (
                86400 * (Qnet_test[:, :, 1, :, :] / (1025 * 4000 * mld_test[:, :, 1, :, :])))) + beta * (86400 * (
            Qnet_test[:, :, 2, :, :] / (1025 * 4000 * mld_test[:, :, 2, :, :])))
                                           ) + beta * (86400 * (
            Qnet_test[:, :, 3, :, :] / (1025 * 4000 * mld_test[:, :, 3, :, :])))) + beta * (
                         (86400 * (Qnet_test[:, :, 4, :, :] / (1025 * 4000 * mld_test[:, :, 4, :, :]))))) + beta * (
                            86400 * (Qnet_test[:, :, 5, :, :] / (1025 * 4000 * mld_test[:, :, 5, :, :])))) + beta * (
                   86400 * (Qnet_test[:, :, 6, :, :] / (1025 * 4000 * mld_test[:, :, 6, :, :])))

ZAdv = aerfa * (aerfa * (aerfa * (aerfa * (
            aerfa * (aerfa * ((u_test[:, :, 0, :, :] * dT_x1)) + beta * (u_test[:, :, 1, :, :] * dT_x2)) + beta * (
                u_test[:, :, 2, :, :] * dT_x3)) + beta * (u_test[:, :, 3, :, :] * dT_x4)) + beta * (
                                     u_test[:, :, 4, :, :] * dT_x5)
                         ) + beta * (u_test[:, :, 5, :, :] * dT_x6)) + beta * ((u_test[:, :, 6, :, :] * dT_x7))

MAdv = aerfa * (aerfa * (aerfa * (aerfa * (
            aerfa * (aerfa * ((v_test[:, :, 0, :, :] * dT_y1)) + beta * (v_test[:, :, 1, :, :] * dT_y2)) + beta * (
                v_test[:, :, 2, :, :] * dT_y3)) + beta * (v_test[:, :, 3, :, :] * dT_y4)) + beta * (
                                     v_test[:, :, 4, :, :] * dT_y5)
                         ) + beta * (v_test[:, :, 5, :, :] * dT_y6)) + beta * ((v_test[:, :, 6, :, :] * dT_y7))

VAdv = aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (aerfa * (
(u_d_test[:, :, 0, :, :] * (sst1_test[:, :, 0, :, :] - T_d_test[:, :, 0, :, :]) / mld_test[:, :, 0, :, :])) + beta * (
                                                                u_d_test[:, :, 1, :, :] * (
                                                                    sst1_test[:, :, 1, :, :] - T_d_test[:, :, 1, :,
                                                                                               :]) / mld_test[:, :, 1,
                                                                                                     :, :])
                                                    ) + beta * (u_d_test[:, :, 2, :, :] * (
            sst1_test[:, :, 2, :, :] - T_d_test[:, :, 2, :, :]) / mld_test[:, :, 2, :, :])) + beta * (
                                              u_d_test[:, :, 3, :, :] * (
                                                  sst1_test[:, :, 3, :, :] - T_d_test[:, :, 3, :, :]) / mld_test[:, :,
                                                                                                        3, :, :])
                                  ) + beta * (u_d_test[:, :, 4, :, :] * (
            sst1_test[:, :, 4, :, :] - T_d_test[:, :, 4, :, :]) / mld_test[:, :, 4, :, :])) + beta * (
                            u_d_test[:, :, 5, :, :] * (sst1_test[:, :, 5, :, :] - T_d_test[:, :, 5, :, :]) / mld_test[:,
                                                                                                             :, 5, :,
                                                                                                             :])) + beta * (
                   u_d_test[:, :, 6, :, :] * (sst1_test[:, :, 6, :, :] - T_d_test[:, :, 6, :, :]) / mld_test[:, :, 6, :,
                                                                                                    :])

Tend = Qnet - ZAdv - MAdv - VAdv

Qnet_1 = Qnet.cpu().detach().numpy()
ZAdv_1 = ZAdv.cpu().detach().numpy()
MAdv_1 = MAdv.cpu().detach().numpy()
VAdv_1 = VAdv.cpu().detach().numpy()
Tend1 = Tend.cpu().detach().numpy()

Tend_true_1 = Tend_true.cpu().detach().numpy()
print('Tend_no_R:{}'.format(np.mean(Tend1)))

print('Qnet:{}'.format(np.mean(Qnet_1)))

print('ZAdv:{}'.format(np.mean(ZAdv_1)))

print('MAdv:{}'.format(np.mean(MAdv_1)))

print('VAdv:{}'.format(np.mean(VAdv_1)))

print('Tend_true:{}'.format(np.mean(Tend_true_1)))
