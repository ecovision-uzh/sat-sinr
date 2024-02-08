import torch
import hydra
import wandb
import json
import os
import pandas as pd
import numpy as np
import pytorch_lightning as pl
#from utils import RandomEuropeDS
from sklearn.metrics import roc_auc_score

class ResidLayer(torch.nn.Module):
    """Residual block used in SINR_Net"""
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        b = self.layers(x)
        x = x + b
        return x

class SINR_Net(torch.nn.Module):
    """Base SINR net"""
    def __init__(self, input_len=4, hidden_dim = 256, dropout = 0.5, layers = 4):
        super().__init__()
        
        self.location_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_len, hidden_dim),
            torch.nn.ReLU(),
            *[ResidLayer(hidden_dim, dropout) for i in range(layers)]
        )
        
        self.classifier = self.net = torch.nn.Linear(hidden_dim, 10040)
        
    def forward(self, x):
        x = self.location_encoder(x)
        x = self.classifier(x)
        return x


class SINR(pl.LightningModule):
    """Base SINR, including metric calculations used in all models"""
    def __init__(self, params, dataset, **kwargs):
        super().__init__(**kwargs)
        
        input_len = 0
        if "loc" in params.dataset.predictors:
            input_len += 4
        if "env" in params.dataset.predictors:
            input_len += 20
        
        self.params = params
        self.predictors = params.dataset.predictors
        if params.model == "log_reg":
            self.net = torch.nn.Linear(input_len, 10040)
        elif params.model == "sinr":
            self.net = SINR_Net(input_len, hidden_dim = params.sinr_hidden, dropout = params.dropout, layers = params.sinr_layers)
        self.dataset = dataset
        self.val_data = dataset.val_data
        self.cutoff = 0.5 
        
        """if "env" in params.dataset.predictors:
            context_feats = dataset.raster"""
        
        self.save_hyperparameters(ignore=['sent2_net', 'dataset', 'random_europe_ds'])
        
    def forward(self, x):
        # Very unclean, but taking care of other predictor combinations in forward, not dataset
        # TODO: Make clean
        enc = x
        if len(x.shape) == 1:
            if not "env" in self.predictors:
                enc = x[:4]
            elif not "loc" in self.predictors:
                enc = x[4:]
        else:
            if not "env" in self.predictors:
                enc = x[:, :4]
            elif not "loc" in self.predictors:
                enc = x[:, 4:]
        return self.net(enc)
    
    def apply_model_and_an_full_loss(self, batch, dataset, params):
        loc_features, labels = batch
        random_loc_features = dataset.sample_encoded_locs(len(batch)).to(labels.device)
        
        loc_pred = torch.sigmoid(self(loc_features))
        rand_pred = torch.sigmoid(self(random_loc_features))

        inds = torch.arange(len(labels))

        loss_pos = -torch.log((1 - loc_pred) + 1e-5)
        loss_bg = -torch.log((1 - rand_pred) + 1e-5)
        loss_pos[inds, labels] = params.pos_weight * -torch.log(loss_pos[inds, labels] + 1e-5)

        return loss_pos.mean() + loss_bg.mean()

    def training_step(self, batch, batch_nb):
        loss = self.apply_model_and_an_full_loss(batch, self.dataset, self.params)
        loss_detached = loss.detach().cpu()
        log_dict = {"train_loss": loss_detached}
        self.log_dict(log_dict, batch_size=len(batch))
        return {"loss": loss, "progress_bar": float(loss_detached)}
    
    def _get_pred_from_key(self, key, device):
        splt_key = key.split("/")
        lon = float(splt_key[0])
        lat = float(splt_key[1])
        x = self.dataset.encode(lon, lat).to(device)
        if not "loc" in self.predictors:
            enc = x[4:]
        else:
            enc = x
        return self.net(enc)
    
    def _calculate_val_metrics(self, log_dict, device):
        
        if self.params.model.startswith("sat_sinr_mf_"):
            for i in range(len(self.net.net.resid_l)):
                log_dict["embedder_stats/" + str(i) + "_" + "weight_mean"] = self.net.net.resid_l[i].embedder.weight.abs().mean()
                log_dict["embedder_stats/" + str(i) + "_" + "weight_std"] = self.net.net.resid_l[i].embedder.weight.abs().std()
                log_dict["embedder_stats/" + str(i) + "_" + "bias_mean"] = self.net.net.resid_l[i].embedder.bias.abs().mean()
                log_dict["embedder_stats/" + str(i) + "_" + "bias_std"] = self.net.net.resid_l[i].embedder.bias.abs().std()
        
        micro_f1_cutoff = 0
        micro_f1_top20 = 0
        micro_f1_top30 = 0
        micro_f1_top25 = 0
        PP_cutoff = 0
        preds = []
        labels = []
        for key in self.val_data.keys():
            pred = self._get_pred_from_key(key, device)
            top_30 = torch.zeros(10040).to(device)
            indics = torch.topk(pred, 30).indices
            top_30[indics] = 1
            above_cutoff = (torch.sigmoid(pred) > self.cutoff).int()
            PP_cutoff += above_cutoff.sum()
            occs = torch.zeros(10040).to(device)
            occs[self.val_data[key]] = 1

            TP = (top_30 * occs).sum()
            FP = (top_30 * (1 - occs)).sum()
            FN = ((1 - top_30) * occs).sum()
            if TP > 0:
                micro_f1_top30 += TP / (TP + (FP + FN) / 2)
            else:
                micro_f1_top30 += 0 

            """top_30 = torch.zeros(10040, dtype=torch.bfloat16).to(device)
            top_30[indics] = pred[indics]"""
            preds.append(pred)
            labels.append(occs)
                
        preds = torch.stack(preds).to("cpu")
        labels = torch.stack(labels).to("cpu")
        sums = labels.sum(dim=0) != 0
        labels = labels[:, sums]
        preds = preds[:, sums]
        log_dict["roc_auc_score_macro"] = roc_auc_score(labels, preds.type(torch.float), average="macro")
        log_dict["roc_auc_score_weighted"] = roc_auc_score(labels, preds.type(torch.float), average="weighted")
        
        log_dict["micro_f1_cutoff"] = micro_f1_cutoff/len(self.val_data)
        log_dict["micro_f1_top30"] = micro_f1_top30/len(self.val_data)
        log_dict["predicted_present_cutoff"] = PP_cutoff/len(self.val_data)
        
        return log_dict
    
    def validation_step(self, batch, batch_nb):
        device = batch[1].device
        
        loss = self.apply_model_and_an_full_loss(batch, self.dataset, self.params)
        loss_detached = loss.detach().cpu()
        log_dict = {"val_loss": loss_detached}
        
        if batch_nb == 0:
            self._calculate_val_metrics(log_dict, device)
                   
        self.log_dict(log_dict, batch_size=len(batch))
        return {"loss": loss, "progress_bar": float(loss_detached)}
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.params.lr,
                              weight_decay = self.params.l2_dec)
        return opt

    
class SAT_SINR(SINR):
    """Abstract Sat-Sinr with adapted loss """
    def __init__(self, params, dataset, sent2_net, **kwargs):
        super().__init__(params, dataset, **kwargs)
        self.net = get_model(params, params.model)
        self.net.sent2_net = sent2_net
        self.dataset = dataset
        self.mode = None
        
        if params.dataset.use_ds_samples:
            reds = dataset
        else:
            reds = RandomEuropeDS(dataset, params.local.bioclim_path, use_ds_samples=False)
        self.reds = torch.utils.data.DataLoader(reds, shuffle=True, batch_size=params.dataset.batchsize, num_workers=params.dataset.num_workers)
        self.random_europe_ds = iter(self.reds)
    
    def set_mode(self, mode=None):
        if mode == "sat_only":
            for n, p in model.net.named_parameters():
                p.requires_grad = False
            for n, p in model.net.sent2_net.named_parameters():
                p.requires_grad = True
        elif mode == "sinr_only":
            for n, p in model.net.named_parameters():
                p.requires_grad = True
            for n, p in model.net.sent2_net.named_parameters():
                p.requires_grad = False
        else:
            for n, p in model.net.named_parameters():
                p.requires_grad = True
    
    def apply_model_and_an_full_loss(self, batch, dataset, params):
        loc_features, sent2_images, labels = batch
        #random_loc_features = dataset.sample_encoded_locs(len(batch)).to(labels.device)
        #rand_pred = torch.sigmoid(self.net(random_loc_features))
        try:
            random_loc_features, random_sent2, _ = next(self.random_europe_ds)
        except:
            self.random_europe_ds = iter(self.reds)
            random_loc_features, random_sent2, _ = next(self.random_europe_ds)
            
        rand_pred = torch.sigmoid(self.net((random_loc_features.to(loc_features.device), random_sent2.to(loc_features.device)), no_sent2=False))
        loc_pred = torch.sigmoid(self.net((loc_features, sent2_images), no_sent2 = False))

        inds = torch.arange(len(labels))

        loss_pos = -torch.log((1 - loc_pred) + 1e-5)
        loss_bg = -torch.log((1 - rand_pred) + 1e-5)
        loss_pos[inds, labels] = params.pos_weight * -torch.log(loss_pos[inds, labels] + 1e-5)

        return loss_pos.mean() + loss_bg.mean()
    
    def _get_pred_from_key(self, key, device):
        splt_key = key.split("/")
        lon = float(splt_key[0])
        lat = float(splt_key[1])
        patchID = splt_key[3]
        sent2 = self.dataset.get_gbif_sent2(patchID).to(device)
        return self.net((self.dataset.encode(lon, lat).to(device), sent2), no_sent2=False)

    
class SASI_LF(torch.nn.Module):
    """Late fusion Sat-SINR that can also be used as sat-only"""
    def __init__(self, params, sat_only=False):
        super().__init__()
        self.net = SINR_Net(24, hidden_dim = params.sinr_hidden, dropout = params.dropout, layers = params.sinr_layers)
        self.sent2_to_classes = torch.nn.Linear(256, 10040)
        self.sat_only = sat_only
        
    def forward(self, x, no_sent2 = True):
        if no_sent2:
            return self.net(x)
        enc, sent2 = x
        sent2_enc = self.sent2_net(sent2)
        sat_classes = self.sent2_to_classes(sent2_enc)
        if sat_classes.shape[0] == 1:
            sat_classes = sat_classes.view(10040)
        if self.sat_only:
            sat_classes
        return self.net(enc) + sat_classes


class ContextResidLayer(torch.nn.Module):
    """Residual layer including context information for middle fusion with various initializations for the merging layer"""
    def __init__(self, hidden_dim, dropout, mode=""):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.mode = mode
        self.sent2hidden = torch.nn.Linear(256, hidden_dim, bias=False)
        if mode.startswith("zero_conv"):
            self.embedder = torch.nn.Linear(hidden_dim, hidden_dim)
            if mode.endswith("normal"):
                self.embedder.weight.detach().normal_()
                self.embedder.bias.detach().normal_()
            elif mode.endswith("zero"):
                self.embedder.weight.detach().zero_()
                self.embedder.bias.detach().zero_()
            elif mode.endswith("xavier_uni"):
                torch.nn.init.xavier_uniform_(self.embedder.weight)
                self.embedder.bias.detach().zero_()
            elif mode.endswith("xavier_normal"):
                torch.nn.init.xavier_normal_(self.embedder.weight)
                torch.nn.init.xavier_normal_(self.embedder.bias)
            elif mode.endswith("kaiming_normal"):
                torch.nn.init.kaiming_normal_(self.embedder.weight)
                torch.nn.init.kaiming_normal_(self.embedder.bias)
            elif mode.endswith("kaiming_uni"):
                torch.nn.init.kaiming_uniform_(self.embedder.weight)
                self.embedder.bias.detach().zero_()
        elif mode == "attn":
            self.embedder = torch.nn.MultiheadAttention(hidden_dim,1)
        else:
            self.embedder = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        
    def forward(self, x, c=None):
        if c is None:
            if self.layers[0].bias.device != x.device:
                self.to(x.device)
            b = self.layers(x)
            x = x + b
        else:
            c = self.sent2hidden(c)
            assert x.shape == c.shape
            if self.layers[0].bias.device != x.device:
                self.to(x.device)
            if self.mode.startswith("zero_conv"):
                b = self.layers(x)
                c_emb = b + self.embedder(c)
            elif self.mode == "attn":
                b = self.layers(x)
                if len(c.shape) == 1:
                    c = c.view(1, -1)
                    b = b.view(1, -1)
                    c_emb = b.flatten() + self.embedder(c, c, b)[0].flatten()
                else:
                    c_emb = b + self.embedder(c, c, b)[0]
            else:
                b = self.layers(x)
                c_emb = self.embedder(torch.cat((b, c), dim=-1))
            x = x + c_emb
        return x

    
class Context_SINR_Net(torch.nn.Module):
    """Sinr but using context vector"""
    def __init__(self, input_len=4, hidden_dim = 256, dropout = 0.5, layers = 4, mode=""):
        super().__init__()
        
        self.inp_l = torch.nn.Linear(input_len, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.resid_l = torch.nn.Sequential(*[ContextResidLayer(hidden_dim, dropout, mode) for i in range(layers)])
        
        self.classifier = self.net = torch.nn.Linear(hidden_dim, 10040)
        
    def forward(self, x, c=None):
        x = self.inp_l(x)
        x = self.relu(x)
        for i in range(len(self.resid_l)):
            x = self.resid_l[i](x, c)
        x = self.classifier(x)
        return x

class SASI_MF(torch.nn.Module):
    """Only SINR adaption to not use SINR_Net. Has various options use different parameter combinations and encoding mechanisms"""
    def __init__(self, params, mode=""):
        super().__init__()
        # nearly all special combinations of predictors outside of the base 3 and sat only is handled here through dataset.predictors
        inp_size = 0
        if "loc" in params.dataset.predictors:
            inp_size += 4
        if "env" in params.dataset.predictors:
            inp_size += 20
        self.net = Context_SINR_Net(inp_size, hidden_dim = params.sinr_hidden, dropout = params.dropout, layers = params.sinr_layers, mode=mode)
        self.predictors = params.dataset.predictors
        
    def forward(self, x, no_sent2 = True):
        if no_sent2:
            if not "env" in self.predictors:
                enc = x[:4]
            elif not "loc" in self.predictors:
                enc = x[4:]
            return self.net(enc)
        
        enc, sent2 = x
        if len(enc.shape) == 1:
            if not "env" in self.predictors:
                enc = enc[:4]
            elif not "loc" in self.predictors:
                enc = enc[4:]
        else:
            if not "env" in self.predictors:
                enc = enc[:,:4]
            elif not "loc" in self.predictors:
                enc = enc[:,4:]
        if no_sent2:
            return self.net(enc)
        sent2_enc = self.sent2_net(sent2)
        if sent2_enc.shape[0] == 1:
            sent2_enc = sent2_enc.view(sent2_enc.shape[1])
        return self.net(enc, c=sent2_enc)
    
class SASI_EF(torch.nn.Module):
    """Early fusion Sat-SINR. The most straightforward of the three."""
    def __init__(self, params, enc_dim=24):
        super().__init__()
        self.enc_dim = enc_dim
        self.net = SINR_Net(24 + enc_dim, hidden_dim = params.sinr_hidden, dropout = params.dropout, layers = params.sinr_layers)
        self.sent2_to_input = torch.nn.Linear(256, enc_dim)
        
    def forward(self, x, no_sent2 = True):
        if no_sent2:
            return self.net(torch.cat((x, torch.zeros(x.shape[0],self.enc_dim).to(x.device)), dim=-1))
        enc, sent2 = x
        sent2_enc = self.sent2_net(sent2)
        sent_input = self.sent2_to_input(sent2_enc)
        if sent_input.shape[0] == 1:
            sent_input = sent_input.view(sent_input.shape[1])
        return self.net(torch.cat((enc, sent_input), dim=-1))
    
def get_model(params, model):
        """Function to get model based on parameter"""
        if model == "sat_sinr_lf":
            net = SASI_LF(params)
        elif model == "sat_only":
            net = SASI_LF(params, sat_only=True)
        elif model == "sat_sinr_ef":
            net = SASI_EF(params)
        elif model == "sat_sinr_mf":
            net = SASI_MF(params)
        elif model == "sat_sinr_mf_zc":
            net = SASI_MF(params, mode="zero_conv_zero")
        elif model == "sat_sinr_mf_zc_n":
            net = SASI_MF(params, mode="zero_conv_normal")
        elif model == "sat_sinr_mf_zc_xu":
            net = SASI_MF(params, mode="zero_conv_xavier_uni")
        elif model == "sat_sinr_mf_zc_xn":
            net = SASI_MF(params, mode="zero_conv_xavier_normal")
        elif model == "sat_sinr_mf_zc_kn":
            net = SASI_MF(params, mode="zero_conv_kaiming_normal")
        elif model == "sat_sinr_mf_zc_ku":
            net = SASI_MF(params, mode="zero_conv_kaiming_uni")
        elif model == "sat_sinr_mf_zc_u":
            net = SASI_MF(params, mode="zero_conv")
        elif model == "sat_sinr_mf_attn":
            net = SASI_MF(params, mode="attn")
        else:
            raise NotImplementeError
        return net