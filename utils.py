import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import v2
import os

def bilinear_interpolate(loc_ip, data):
    """bilinear interpolation for raster data"""
    # Taken from: https://github.com/elijahcole/sinr/blob/main/utils.py
    
    # loc is N x 2 vector, where each row is [lon,lat] entry
    #   each entry spans range [-1,1]
    # data is H x W x C, height x width x channel data matrix
    # op will be N x C matrix of interpolated features

    assert data is not None

    # map to [0,1], then scale to data size
    loc = (loc_ip.clone() + 1) / 2.0
    loc[:,1] = 1 - loc[:,1] # this is because latitude goes from +90 on top to bottom while
                            # longitude goes from -90 to 90 left to right

    assert not torch.any(torch.isnan(loc))
    
    # cast locations into pixel space
    loc[:, 0] *= (data.shape[1]-1)
    loc[:, 1] *= (data.shape[0]-1)

    loc_int = torch.floor(loc).long()  # integer pixel coordinates
    xx = loc_int[:, 0]
    yy = loc_int[:, 1]
    xx_plus = xx + 1
    xx_plus[xx_plus > (data.shape[1]-1)] = data.shape[1]-1
    yy_plus = yy + 1
    yy_plus[yy_plus > (data.shape[0]-1)] = data.shape[0]-1

    loc_delta = loc - torch.floor(loc)   # delta values
    dx = loc_delta[:, 0].unsqueeze(1)
    dy = loc_delta[:, 1].unsqueeze(1)

    interp_val = data[yy, xx, :]*(1-dx)*(1-dy) + data[yy, xx_plus, :]*dx*(1-dy) + \
                 data[yy_plus, xx, :]*(1-dx)*dy   + data[yy_plus, xx_plus, :]*dx*dy

    return interp_val


"""def create_submission(model, dataset, tag="default"):
    submission_data = pd.read_csv('/data/jdolli/glc23_data/For_submission/test_blind.csv', sep=";", header='infer', low_memory=False)
    device = "cpu"
    subm = open("./submission_" + tag + ".csv", "w")
    subm.write("Id,Predicted\n")
    mc = torch.zeros(10040)
    mc[most_common] = 1
    #for ent in tqdm(submission_data.iterrows()):
    for ent in submission_data.iterrows():
        idx, data_dict= ent
        lon = data_dict[["lon"]].to_numpy()[0]
        lat = data_dict[["lat"]].to_numpy()[0]
        with torch.no_grad():
            pred = model(dataset.encode(lon, lat).to(device)) * mc
        top_25 = torch.topk(pred, 25).indices
        str_list = [str(idx) for idx in sorted(top_25.numpy())]
        subm.write(str(data_dict[["Id"]].to_numpy()[0]) + "," + " ".join(str_list) + "\n")
        
    subm.close()"""

    
"""class RandomEuropeDS(torch.utils.data.Dataset):
    def __init__(self, dataset, bioclim_path = "undefined",
                 europe_img_path = "/data/jdolli/sentinel_2 2021 Europe/", use_ds_samples = False):
        super().__init__()
        RES_LON = 502
        RES_LAT = 408
        max_lon = 34.55792
        min_lon = -10.53904
        max_lat = 71.18392
        min_lat = 34.56858
        self.locs = []
        if use_ds_samples:
            for i in range(RES_LON):
                for j in range(RES_LAT):
                    lon = i/RES_LON
                    lat = j/RES_LAT
                    lon = lon * (max_lon - min_lon) + min_lon
                    lat = lat * (max_lat - min_lat) + min_lat
                    self.locs.append((lon, lat))
        else:
            paths = os.listdir(europe_img_path + "random_rgb")
            for path in paths:
                lat = path.split(",")[0]
                lon = path.split(",")[1][:-5]
                self.locs.append((float(lon), float(lat)))
                
        self.use_ds_samples = use_ds_samples
        self.bioclim_path = bioclim_path
        self.europe_img_path = europe_img_path
        self.dataset = dataset
        self.to_tensor = torchvision.transforms.PILToTensor()
        self.failure_counter = 0
        
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])
                
    def __len__(self):
        return len(self.locs)
        
    def __getitem__(self, idx):
        lon, lat = self.locs[idx]
        loc_env = self.dataset.encode(lon, lat)
        pos = str(lat) + "," + str(lon)
        if self.use_ds_samples:
            rgb_path = self.europe_img_path + "rgb/" + pos + ".jpeg"
            nir_path = self.europe_img_path + "nir/" + pos + ".jpeg"
        else:
            rgb_path = self.europe_img_path + "random_rgb/" + pos + ".jpeg"
            nir_path = self.europe_img_path + "random_nir/" + pos + ".jpeg"                           
        try:
            rgb = Image.open(rgb_path)
            nir = Image.open(nir_path)
            sent2 = torch.concat([self.to_tensor(rgb), self.to_tensor(nir)], dim=0)/255
        except:
            sent2 = torch.zeros(4, 128, 128)
            self.failure_counter += 1
        return loc_env, self.transforms(sent2), 0"""
        

class DummyParams():
    def __init__(self):
        pass

class DefaultParams():
    def __init__(self, sinr):
        self.dataset = DummyParams()
        self.local = DummyParams()
        
        self.pos_weight = 2048
        self.lr = 5e-4
        self.l2_dec = 0
        self.epochs = 7
        self.model = "sinr"
        self.dataset.predictors = "loc"
        self.sinr_layers = 8
        self.sinr_hidden = 512
        self.dropout = 0.3
        self.tag = "Jupyter"
        self.embedder = "ae_default"
        self.dataset.batchsize = 2048
        self.dataset.use_ds_samples = True
        self.dataset.num_workers = 16
        self.local.sent_data_path = "/shares/wegner.ics.uzh/glc23_data/SatelliteImages/"
        self.local.bioclim_lr_path = "/shares/wegner.ics.uzh/glc23_data/sinr_data/data/env/bioclim_elevation_scaled_europe.npy"
        self.local.bioclim_path = "/shares/wegner.ics.uzh/glc23_data/bioclim+elev/bioclim_elevation_scaled_europe.npy"
        self.local.dataset_file_path = "/shares/wegner.ics.uzh/glc23_data/Presences_only_train.csv"
        self.local.cp_dir_path = "/scratch/jribas/sent-sinr/checkpoints"
        self.local.logs_dir_path = "/scratch/jribas/sent-sinr/wandb_logs"
        self.local.val_data_path = "/shares/wegner.ics.uzh/glc23_data/Presence_Absence_surveys/Presences_Absences_train.csv"
        self.local.gpu = False
        
        if sinr:
            self.dataset.predictors = "loc_env"  # "loc"
            self.epochs = 15
            self.pos_weight = 2048
            self.model = "sinr"
