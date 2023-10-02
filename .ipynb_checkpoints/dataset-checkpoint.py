import json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from utils import bilinear_interpolate

#load data and put it into torch dataset
class SINR_DS(torch.utils.data.Dataset):
    def __init__(self, dataset_file, predictors, bioclim_path = "/data/jdolli/glc23_data/sinr_data/data/env/bioclim_elevation_scaled.npy",
                 sent_data_path = "/data/jdolli/glc23_data/SatelliteImages/"):
        super().__init__()
        #self.data = pd.read_csv(dataset_file, sep=";", header='infer', low_memory=False)
        self.data = dataset_file
        with open('/data/jdolli/glc23_data/Presence_Absence_surveys/loc_to_spec.csv', "r") as f:
            self.val_data = json.load(f)
        self.predictors = predictors
        if "sent2" in predictors:
            self.to_tensor = PILToTensor()
        if "env" in predictors:
            context_feats = np.load(bioclim_path).astype(np.float32)
            self.raster = torch.from_numpy(context_feats)
            self.raster[torch.isnan(self.raster)] = 0.0 # replace with mean value (0 is mean post-normalization)
            
        self.sent_data_path = sent_data_path
        
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
        return bilinear_interpolate(torch.tensor([[lon/90, lat/90]]), self.raster)
    
    def get_loc_env(self, lon, lat):
        lon_norm, lat_norm = self._normalize_loc_to_uniform(lon, lat)
        loc_enc = torch.tensor(self._encode_loc(lon_norm, lat_norm), dtype=torch.float32)
        env_enc = self.get_env_raster(lon, lat).type("torch.FloatTensor")
        return torch.cat((loc_enc, env_enc.view(20)))
    
    def encode(self, lon, lat):
        if "loc_env" in self.predictors:
            return self.get_loc_env(lon, lat)
        else:
            lon_norm, lat_norm = self._normalize_loc_to_uniform(lon, lat)
            return torch.tensor(self._encode_loc(lon_norm, lat_norm), dtype=torch.float32)
        
    def get_gbif_sent2(self, pid):
        rgb_path = self.sent_data_path + "rgb/" + str(pid)[-2:] + "/" + str(pid)[-4:-2]+ "/" + str(pid) + ".jpeg"
        nir_path = self.sent_data_path + "nir/" + str(pid)[-2:] + "/" + str(pid)[-4:-2]+ "/" + str(pid) + ".jpeg"
        rgb = Image.open(rgb_path)
        nir = Image.open(nir_path)
        return torch.concat([self.to_tensor(rgb), self.to_tensor(nir)], dim=0)/255
    
    def __getitem__(self, idx):
        data_dict = self.data.iloc[idx]
        lon, lat = tuple(data_dict[["lon", "lat"]].to_numpy())
        if "sent2" in self.predictors:
            return self.encode(lon, lat), self.get_gbif_sent2(data_dict["patchID"]), torch.tensor(data_dict["speciesId"])
        else:
            return self.encode(lon, lat), torch.tensor(data_dict["speciesId"])