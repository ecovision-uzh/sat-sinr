import torch
import pytorch_lightning as pl
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

    def __init__(self, input_len=4, hidden_dim=256, dropout=0.5, layers=4):
        super().__init__()

        self.location_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_len, hidden_dim),
            torch.nn.ReLU(),
            *[ResidLayer(hidden_dim, dropout) for i in range(layers)]
        )

        self.classifier = torch.nn.Linear(hidden_dim, 10040)

    def forward(self, x):
        x = self.location_encoder(x)
        x = self.classifier(x)
        return x


class SINR(pl.LightningModule):
    """Base SINR, including metric calculations used in all models.
    Also includes the log_reg implementation, replacing the SINR_net with a single layer."""

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
            self.net = SINR_Net(input_len, hidden_dim=params.sinr_hidden, dropout=params.dropout,
                                layers=params.sinr_layers)
        self.dataset = dataset
        self.test_data = dataset.test_data

        self.max_macro_roc_auc = 0
        self.max_weighted_roc_auc = 0
        self.max_micro_f1 = 0

        self.save_hyperparameters(ignore=['dataset'])

    def forward(self, x):
        return self.net(x)

    def apply_model_and_an_full_loss(self, batch, dataset, params):
        """Get x, sample random background samples, process both through the network, and calculate the loss."""
        loc_features, labels = batch
        random_loc_features = dataset.sample_encoded_locs(len(loc_features)).to(labels.device)

        loc_pred = torch.sigmoid(self(loc_features))
        rand_pred = torch.sigmoid(self(random_loc_features))

        assert len(rand_pred) == len(loc_pred)

        inds = torch.arange(len(labels))

        # Assume all classes to be absent
        loss_pos = -torch.log((1 - loc_pred) + 1e-5)
        # Assume all classes at the random background locations to be absent
        loss_bg = -torch.log((1 - rand_pred) + 1e-5)
        # For the confirmed occurrences, switch the sign of the predicted probability and upscale with pos_weight
        loss_pos[inds, labels] = params.pos_weight * -torch.log(loc_pred[inds, labels] + 1e-5)

        return loss_pos.mean() + loss_bg.mean()

    def training_step(self, batch, batch_nb):
        """Single train step on a batch."""
        loss = self.apply_model_and_an_full_loss(batch, self.dataset, self.params)
        loss_detached = loss.detach().cpu()
        log_dict = {"train_loss": loss_detached}
        self.log_dict(log_dict, batch_size=len(batch))
        return {"loss": loss, "progress_bar": float(loss_detached)}

    def _get_pred_from_key(self, key, device):
        """Get the predictors and model ouputs for a test_data key."""
        splt_key = key.split("/")
        lon = float(splt_key[0])
        lat = float(splt_key[1])
        x = self.dataset.encode(lon, lat).to(device)
        return self.net(x)

    def _calculate_test_metrics(self, log_dict, device):
        """Iterate over the whole test_data to calculate the scores."""
        micro_f1_top30 = 0
        preds = []
        labels = []
        for key in self.test_data.keys():
            pred = self._get_pred_from_key(key, device)
            top_30 = torch.zeros(10040).to(device)
            # Consider the 30 classes with the highest predicted probability to be present
            indics = torch.topk(pred, 30).indices
            top_30[indics] = 1
            occs = torch.zeros(10040).to(device)
            occs[self.test_data[key]] = 1

            # Calculation of true positives and co. for Micro-F1 calc
            TP = (top_30 * occs).sum()
            FP = (top_30 * (1 - occs)).sum()
            FN = ((1 - top_30) * occs).sum()
            if TP > 0:
                micro_f1_top30 += TP / (TP + (FP + FN) / 2)
            else:
                micro_f1_top30 += 0

            preds.append(pred)
            labels.append(occs)

        preds = torch.stack(preds).to("cpu")
        labels = torch.stack(labels).to("cpu")
        # Reduce labels and preds to only those classes that appear in the test_data
        sums = labels.sum(dim=0) != 0
        labels = labels[:, sums]
        preds = preds[:, sums]
        micro_f1 = micro_f1_top30 / len(self.test_data)
        macro_roc_auc = roc_auc_score(labels, preds, average="macro")
        weighted_roc_auc = roc_auc_score(labels, preds, average="weighted")
        log_dict["roc_auc_score_macro"] = macro_roc_auc
        log_dict["roc_auc_score_weighted"] = weighted_roc_auc
        log_dict["micro_f1_top30"] = micro_f1

        if micro_f1 > self.max_micro_f1:
            self.max_micro_f1 = micro_f1
        if macro_roc_auc > self.max_macro_roc_auc:
            self.max_macro_roc_auc = macro_roc_auc
        if weighted_roc_auc > self.max_weighted_roc_auc:
            self.max_weighted_roc_auc = weighted_roc_auc

        log_dict["max_roc_auc_score_macro"] = self.max_macro_roc_auc
        log_dict["max_roc_auc_score_weighted"] = self.max_weighted_roc_auc
        log_dict["max_micro_f1_top30"] = self.max_micro_f1

        return log_dict

    def validation_step(self, batch, batch_nb):
        """Same as train, except also calculating metrics on the test_data once per epoch."""
        device = batch[1].device

        loss = self.apply_model_and_an_full_loss(batch, self.dataset, self.params)
        loss_detached = loss.detach().cpu()
        log_dict = {"val_loss": loss_detached}

        if batch_nb == 0:
            # We calculate test_metrics once in each epoch to track the change of performance throughout training
            self._calculate_test_metrics(log_dict, device)

        self.log_dict(log_dict, batch_size=len(batch))
        return {"loss": loss, "progress_bar": float(loss_detached)}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.params.lr,
                               weight_decay=self.params.l2_dec)
        return opt


class SAT_SINR(SINR):
    """Abstract Sat-Sinr with adapted loss """

    def __init__(self, params, dataset, sent2_net, **kwargs):
        super().__init__(params, dataset, **kwargs)
        self.net = get_model(params, params.model)
        self.net.sent2_net = sent2_net
        self.dataset = dataset
        # Instantiate another DataLoader from the dataset to serve as background samples
        self.re_dl = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=params.dataset.batchsize,
                                                 num_workers=params.dataset.num_workers)
        # Instantiate iterator from the dataloader
        self.re_iter = iter(self.re_dl)

    def apply_model_and_an_full_loss(self, batch, dataset, params):
        """Same as in SINR, but due to the Sentinel-2 images being pre-processed, we can't sample randomly across
        Europe. Thus, we clone the training samples as random background samples."""
        loc_features, sent2_images, labels = batch
        try:
            random_loc_features, random_sent2, _ = next(self.re_iter)
        except:
            # If the dataloader is empty, restock
            self.re_iter = iter(self.re_dl)
            random_loc_features, random_sent2, _ = next(self.re_iter)

        rand_pred = torch.sigmoid(
            self.net((random_loc_features.to(loc_features.device), random_sent2.to(loc_features.device))))
        loc_pred = torch.sigmoid(self.net((loc_features, sent2_images)))

        # Make sure that all have the same length (Avoiding the edge-case of last batch in dl being smaller than rest).
        rand_pred = rand_pred[:len(loc_pred)]
        loc_pred = loc_pred[:len(rand_pred)]
        labels = labels[:len(loc_pred)]

        inds = torch.arange(len(labels))

        loss_pos = -torch.log((1 - loc_pred) + 1e-5)
        loss_bg = -torch.log((1 - rand_pred) + 1e-5)
        loss_pos[inds, labels] = params.pos_weight * -torch.log(loc_pred[inds, labels] + 1e-5)

        return loss_pos.mean() + loss_bg.mean()

    def _get_pred_from_key(self, key, device):
        """Same as SINR, but also loading the Sentinel-2 image."""
        splt_key = key.split("/")
        lon = float(splt_key[0])
        lat = float(splt_key[1])
        patchID = splt_key[3]
        sent2 = self.dataset.get_gbif_sent2(patchID).to(device)
        return self.net((self.dataset.encode(lon, lat).to(device), sent2))


class SASI_LF(torch.nn.Module):
    """Late fusion Sat-SINR that can also be used as sat-only."""

    def __init__(self, params, sat_only=False):
        super().__init__()
        inp_size = 0
        if "loc" in params.dataset.predictors:
            inp_size += 4
        if "env" in params.dataset.predictors:
            inp_size += 20
        self.net = SINR_Net(inp_size, hidden_dim=params.sinr_hidden, dropout=params.dropout, layers=params.sinr_layers)
        self.sent2_to_classes = torch.nn.Linear(256, 10040)
        self.sat_only = sat_only

    def forward(self, x):
        enc, sent2 = x
        sent2_enc = self.sent2_net(sent2)
        sat_classes = self.sent2_to_classes(sent2_enc)
        if sat_classes.shape[0] == 1:
            sat_classes = sat_classes.view(10040)
        if self.sat_only:
            # For sat_only, we only return the embedder output without applying the SINR net
            return sat_classes
        return self.net(enc) + sat_classes


class ContextResidLayer(torch.nn.Module):
    """Residual layer including context information for middle fusion."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.embedder = torch.nn.Linear(256, hidden_dim)
        # Init embedder weights to zero
        self.embedder.weight.detach().zero_()
        self.embedder.bias.detach().zero_()

    def forward(self, x, c):
        """We add in the context information c."""
        b = self.layers(x)
        return x + b + self.embedder(c)


class Context_SINR_Net(torch.nn.Module):
    """Sinr but using context vector that is added in at each layer"""

    def __init__(self, input_len=4, hidden_dim=256, dropout=0.5, layers=4):
        super().__init__()

        self.inp_l = torch.nn.Linear(input_len, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.resid_l = torch.nn.Sequential(*[ContextResidLayer(hidden_dim, dropout) for i in range(layers)])

        self.classifier = self.net = torch.nn.Linear(hidden_dim, 10040)

    def forward(self, x, c):
        """Using context vector c along with input x."""
        x = self.inp_l(x)
        x = self.relu(x)
        for i in range(len(self.resid_l)):
            x = self.resid_l[i](x, c)
        x = self.classifier(x)
        return x


class SASI_MF(torch.nn.Module):
    """Use a context-enabled version of SINR_Net to feed in a context vector in each layer."""

    def __init__(self, params):
        super().__init__()

        inp_size = 0
        if "loc" in params.dataset.predictors:
            inp_size += 4
        if "env" in params.dataset.predictors:
            inp_size += 20
        self.net = Context_SINR_Net(inp_size, hidden_dim=params.sinr_hidden, dropout=params.dropout,
                                    layers=params.sinr_layers)
        self.predictors = params.dataset.predictors

    def forward(self, x):
        # Get both location/environmental embedding and sat from batch
        enc, sent2 = x
        # Get satellite embedding
        sent2_enc = self.sent2_net(sent2)
        if sent2_enc.shape[0] == 1:
            sent2_enc = sent2_enc.view(sent2_enc.shape[1])
        # Pass the satellite embedding to the network as context vector
        return self.net(enc, c=sent2_enc)


class SASI_EF(torch.nn.Module):
    """Early fusion Sat-SINR."""

    def __init__(self, params, enc_dim=24):
        super().__init__()
        inp_size = 0
        if "loc" in params.dataset.predictors:
            inp_size += 4
        if "env" in params.dataset.predictors:
            inp_size += 20
        self.net = SINR_Net(inp_size + enc_dim, hidden_dim=params.sinr_hidden, dropout=params.dropout,
                            layers=params.sinr_layers)
        self.sent2_to_input = torch.nn.Linear(256, enc_dim)

    def forward(self, x):
        """Cat the loc-env and sat encodings into one before passing them into the net."""
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
    else:
        raise NotImplementedError
    return net
