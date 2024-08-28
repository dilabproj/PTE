
import pandas as pd
import dataloader_last as dataloader
import torch

def roads(root_path, road):
    origin_df = pd.read_csv(root_path+'/traffic_data_'+road+'_dataframe.csv')
    trav_mean, trav_std = origin_df['traveltime'].mean(), origin_df['traveltime'].std()
    return origin_df,trav_mean,trav_std     

def inverse_transform(trav_mean, trav_std, series):
    for i in range(len(series)):
        series[i] = series[i] * trav_std + trav_mean # z scaling
    return series  

# data loader
def create_dataloader(road, root_path, selecteddata, n_batch, n_workers=0):
    train_dataset = dataloader.TTPLoader(road,data_path=root_path, mode="train",selecteddata=selecteddata) 
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=n_batch, 
        shuffle=True, 
        num_workers=n_workers
    )
    validation_dataset = dataloader.TTPLoader(road,data_path=root_path,mode="validation",selecteddata=selecteddata) 
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset, 
        batch_size=n_batch, 
        shuffle=True, 
        num_workers=n_workers
    )    
    test_dataset = dataloader.TTPLoader(road,data_path=root_path, mode="test",selecteddata=selecteddata) 
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=n_batch, 
        shuffle=True, 
        num_workers=n_workers
    )
    loaders = {'train':train_loader,'validation':validation_loader, 'test':test_loader}
    return loaders

#
# data_x:(size, length:96, feature:316)

#length: for instance, current time is 20:00			
#previous 7 days  +	 Short-term    ---->	  Predicted time
#20:00-21:00	       19:00-20:00 	            20:00-21:00 
# known              known                     unknown

# feature: [0,1]	    [1,2]	[2,14]	[14,21]	[21,24]	 [24,312]	[312,316]
#          travel time	speed	 month	 week	holiday	  slot	      peak
#          1	          1	      12	  7	       3	  288	       4
