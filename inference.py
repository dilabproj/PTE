import torch
from torch import nn
import numpy as np
from torchmetrics.regression import SymmetricMeanAbsolutePercentageError
import data

# =======================================================================================================
# =======================================================================================================

def transformer_test(model, test_loader, trav_mean, trav_std):
    
    device = next(model.parameters()).device
    mse = nn.MSELoss().to(device)
    smape = SymmetricMeanAbsolutePercentageError().to(device)
    mae = nn.L1Loss().to(device)
    
    y_all = torch.tensor([]).to(device)
    output_all = torch.tensor([]).to(device)
    
    model.eval()
    with torch.no_grad():

        for i, (src, label) in enumerate(test_loader):
            
            # to cuda
            src = src.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            
            # prediction
            final_prediction = model(src, label, evaluate=True)

            # inverse transform
            output = data.inverse_transform(trav_mean, trav_std, final_prediction[:,:,0])
            y = data.inverse_transform(trav_mean, trav_std, label[:,:,0])
            
            # concate results
            y_all = torch.cat((y_all, y), 0)
            output_all = torch.cat((output_all, output), 0)
        
    # compute loss
    all_rmse_loss = mse(output_all, y_all).item() ** 0.5
    all_mae_loss = mae(output_all, y_all).item()
    all_smape_loss = smape(output_all, y_all).item() * 100
            
    return all_mae_loss, all_rmse_loss, all_smape_loss

# =======================================================================================================
# =======================================================================================================
def transformer_test_peak_off_peak(model, test_loader, trav_mean, trav_std):
    
    device = next(model.parameters()).device
    mse = nn.MSELoss().to(device)
    smape = SymmetricMeanAbsolutePercentageError().to(device)
    mae = nn.L1Loss().to(device)
    
    y_all = torch.tensor([]).to(device)
    y_peak_all = torch.tensor([]).to(device)
    y_off_peak_all = torch.tensor([]).to(device)
    output_all = torch.tensor([]).to(device)
    output_peak_all = torch.tensor([]).to(device)
    output_off_peak_all = torch.tensor([]).to(device)
    
    model.eval()
    with torch.no_grad():

        for i, (src, label) in enumerate(test_loader):
            
            # peak and off-peak list
            #peak_list = (np.count_nonzero(src[:,0:12,108:132].numpy(),axis=(1,2))==12) | \
            #            (np.count_nonzero(src[:,0:12,210:258].numpy(),axis=(1,2))==12)
            peak_list= (np.count_nonzero(src[:,0:12,96:120].numpy(),axis=(1,2))==12) | \
                        (np.count_nonzero(src[:,0:12,198:246].numpy(),axis=(1,2))==12)
            off_peak_list = torch.tensor([not item for item in peak_list]).unsqueeze(1).repeat(1, 12).to(device)
            peak_list = torch.tensor(peak_list).unsqueeze(1).repeat(1, 12).to(device)
            
            # to cuda
            src = src.to(torch.float32).to(device)
            label = label.to(torch.float32).to(device)
            
            # prediction
            final_prediction = model(src, label, evaluate=True)

            # inverse transform
            output = data.inverse_transform(trav_mean, trav_std, final_prediction[:,:,0])
            y = data.inverse_transform(trav_mean, trav_std, label[:,:,0])
            
            # Separate the prediction results for peak and off-peak
            peak_y = torch.masked_select(y, peak_list)
            peak_output = torch.masked_select(output, peak_list)
            off_peak_y = torch.masked_select(y, off_peak_list)
            off_peak_output = torch.masked_select(output, off_peak_list)
            
            # concate results
            y_all = torch.cat((y_all, y), 0)
            output_all = torch.cat((output_all, output), 0)
            y_peak_all = torch.cat((y_peak_all, peak_y), 0)
            output_peak_all = torch.cat((output_peak_all, peak_output), 0)
            y_off_peak_all = torch.cat((y_off_peak_all, off_peak_y), 0)
            output_off_peak_all = torch.cat((output_off_peak_all, off_peak_output), 0)
        
    # compute loss
    all_rmse_loss = mse(output_all, y_all).item() ** 0.5
    all_mae_loss = mae(output_all, y_all).item()
    all_smape_loss = smape(output_all, y_all).item() * 100
    peak_rmse_loss = mse(output_peak_all, y_peak_all).item() ** 0.5
    peak_mae_loss = mae(output_peak_all, y_peak_all).item()
    peak_smape_loss = smape(output_peak_all, y_peak_all).item() * 100
    off_peak_rmse_loss = mse(output_off_peak_all, y_off_peak_all).item() ** 0.5
    off_peak_mae_loss = mae(output_off_peak_all, y_off_peak_all).item()
    off_peak_smape_loss = smape(output_off_peak_all, y_off_peak_all).item() * 100
            
    return all_mae_loss, all_rmse_loss, all_smape_loss, \
           peak_mae_loss, peak_rmse_loss, peak_smape_loss, \
           off_peak_mae_loss, off_peak_rmse_loss, off_peak_smape_loss
