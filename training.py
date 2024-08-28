import torch
from torch import nn
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
import numpy as np
import inference
import data



# =======================================================================================================
# =======================================================================================================
def transformer_train(n_epoche, model, train_loader, valid_loader, trav_mean, trav_std, optimizer, scheduler, path, predict_time):
    
    device = next(model.parameters()).device
    mse = nn.MSELoss().to(device)
    smape = SymmetricMeanAbsolutePercentageError().to(device)
    mae = nn.L1Loss().to(device)
    
    best_loss = float('inf')
    for epoch in range(n_epoche):
        train_loss, val_loss, val_rmse_loss, val_smape_loss = [], [], [], []

        model.train()
        for i, (src, tgt) in enumerate(train_loader):

            # to cuda
            src = src.to(torch.float32).to(device)
            tgt = tgt.to(torch.float32).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # prediction
            prediction = model(src, tgt, evaluate=False)

            # Compute and backprop loss
            prediction_y = data.inverse_transform(trav_mean, trav_std, prediction[:,:,0])
            tgt_y = data.inverse_transform(trav_mean, trav_std, tgt[:,:,0])
            mse_loss = mse(prediction_y, tgt_y)
            mse_loss.backward()
            train_loss.append(mse_loss.item())

            # Take optimizer step
            optimizer.step()
        scheduler.step()
            
        # validation
        #val_mae_loss, val_rmse_loss, val_smape_loss = inference.transformer_test(model, valid_loader, trav_mean, trav_std)
        #print("=> train_mse_loss: {:.4f}   val_mae_loss: {:.4f}".format(np.mean(train_loss), val_mae_loss))
        val_mae_loss, val_rmse_loss, val_smape_loss, \
        peak_mae_loss, peak_rmse_loss, peak_smape_loss, \
        off_peak_mae_loss, off_peak_rmse_loss, off_peak_smape_loss = inference.transformer_test_peak_off_peak(model, valid_loader, trav_mean, trav_std)
        print("=> train_mse_loss: {:.4f}   val_mae_loss: {:.4f} val_peak_mae_loss: {:.4f} ".format(np.mean(train_loss), val_mae_loss, peak_mae_loss))

        if val_mae_loss < best_loss:
            best_loss = val_mae_loss
            torch.save(model.state_dict(), path+f'model_{predict_time}.pth')

            