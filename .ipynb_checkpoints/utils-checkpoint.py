import torch
import numpy as np

def bilinear_interpolate(loc_ip, data):
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