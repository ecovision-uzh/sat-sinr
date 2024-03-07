import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import PILToTensor, v2
from utils import bilinear_interpolate

#load data and put it into torch dataset
class SINR_DS(torch.utils.data.Dataset):
    def __init__(self, params, dataset_file, predictors, bioclim_path,
                 sent_data_path, use_subm_val=False):
        super().__init__()
        self.data = dataset_file
        self.bioclim_path = bioclim_path
        
        # val_data is not used by the dataset itself, but the model needs this object
        with open(params.local.val_data_path, "r") as f:
            data_val = pd.read_csv(f, sep=";", header="infer", low_memory=False)
            self.val_data = data_val.groupby(["patchID", "dayOfYear", "lon", "lat"]).agg({"speciesId": lambda x: list(x)}).reset_index()
            self.val_data = {str(entry["lon"]) + "/" + str(entry["lat"]) + "/" + str(entry["dayOfYear"]) + "/" + str(entry["patchID"]): entry["speciesId"] for idx, entry in self.val_data.iterrows()}
            
        self.predictors = predictors
        if "sent2" in predictors:
            self.to_tensor = PILToTensor()
        if "env" in predictors:
            context_feats = np.load(bioclim_path).astype(np.float32)
            self.raster = torch.from_numpy(context_feats)
            self.raster[torch.isnan(self.raster)] = 0.0 # replace with mean value (0 is mean post-normalization)
            
        self.sent_data_path = sent_data_path
        
        self.transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def _normalize_loc_to_uniform(self, lon, lat):
        lon = (lon - (-10.53904)) / (34.55792 - (-10.53904))
        lat = (lat - 34.56858) / (71.18392 - 34.56858)
        return lon, lat
    
    def _encode_loc(self, lon, lat):
        features = [np.sin(np.pi * lon), np.cos(np.pi * lon), np.sin(np.pi * lat), np.cos(np.pi * lat)]
        return np.stack(features, axis=-1)
    
    def sample_encoded_locs(self, size):
        lon = np.random.rand(size)
        lat = np.random.rand(size)
        loc_enc = torch.tensor(self._encode_loc(lon, lat), dtype=torch.float32)
        if "env" in self.predictors:
            lon = torch.tensor(lon * 2 - 1)
            lat = torch.tensor(lat * 2 - 1)
            env_enc = bilinear_interpolate(torch.stack([lon, lat], dim = 1), self.raster)
            return torch.cat([loc_enc, env_enc], dim=1).type("torch.FloatTensor")
        else:
            return loc_enc
    
    def get_env_raster(self, lon, lat):
        if "europe" in self.bioclim_path:
            lat = (lat - 34) / (72-34)
            lon = (lon - (-11)) / (35-(-11))
            lon = lon * 2 - 1
            lat = lat * 2 - 1
        else:
            lon = lon / 180
            lat = lat / 90
        return bilinear_interpolate(torch.tensor([[lon, lat]]), self.raster)
    
    def get_loc_env(self, lon, lat):
        lon_norm, lat_norm = self._normalize_loc_to_uniform(lon, lat)
        loc_enc = torch.tensor(self._encode_loc(lon_norm, lat_norm), dtype=torch.float32)
        env_enc = self.get_env_raster(lon, lat).type("torch.FloatTensor")
        return torch.cat((loc_enc, env_enc.view(20)))
    
    def encode(self, lon, lat):
        if "env" in self.predictors:
            return self.get_loc_env(lon, lat)
        else:
            lon_norm, lat_norm = self._normalize_loc_to_uniform(lon, lat)
            return torch.tensor(self._encode_loc(lon_norm, lat_norm), dtype=torch.float32)
        
    def get_gbif_sent2(self, pid):
        rgb_path = self.sent_data_path + "rgb/" + str(pid)[-2:] + "/" + str(pid)[-4:-2]+ "/" + str(pid) + ".jpeg"
        nir_path = self.sent_data_path + "nir/" + str(pid)[-2:] + "/" + str(pid)[-4:-2]+ "/" + str(pid) + ".jpeg"
        rgb = Image.open(rgb_path)
        nir = Image.open(nir_path)
        img = torch.concat([self.to_tensor(rgb), self.to_tensor(nir)], dim=0)/255
        return self.transforms(img)
    
    def __getitem__(self, idx):
        """ The steps, in which the dataset constructs a datapoint, are a bit convoluted """
        data_dict = self.data.iloc[idx]
        lon, lat = tuple(data_dict[["lon", "lat"]].to_numpy())
        if "sent2" in self.predictors:
            return self.encode(lon, lat), self.get_gbif_sent2(data_dict["patchID"]), torch.tensor(data_dict["speciesId"])
        else:
            return self.encode(lon, lat), torch.tensor(data_dict["speciesId"])

        
def create_datasets(params):
    """Creates dataset and dataloaders from the various files"""
    dataset_file = pd.read_csv(params.local.dataset_file_path, sep=";", header='infer', low_memory=False)
    if params.dataset.predictors.endswith("LR"):
        bioclim_path = params.local.bioclim_lr_path
    else:
        bioclim_path = params.local.bioclim_path
    dataset = SINR_DS(params, dataset_file, params.dataset.predictors, sent_data_path = params.local.sent_data_path, bioclim_path = bioclim_path)
    ds_train, ds_val = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=params.dataset.batchsize, num_workers=params.dataset.num_workers)
    val_loader = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=params.dataset.batchsize, num_workers=params.dataset.num_workers)
    return dataset, train_loader, val_loader