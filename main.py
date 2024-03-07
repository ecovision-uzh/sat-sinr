import wandb
import hydra
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

from embedders import get_embedder
from models import *
from dataset import *

def get_logger(params, tag=""):
    """
    Instantiates Weights and Biases logger
    """
    wandb.finish()
    name = params.model
    if params.model == "sinr" or params.model == "log_reg":
        name += " " + params.dataset.predictors
    elif "sat" in params.model:
        name += " " + params.embedder
    if params.validate:
        name += " val" 
    name += " " + tag
    
    logger = hydra.utils.instantiate({"_target_": "pytorch_lightning.loggers.WandbLogger",
            "name": name,
            "save_dir": params.local.logs_dir_path,
            "project": "sinr_on_glc23"})
    return logger


def train_model(params, dataset, train_loader, val_loader, provide_model=None, logger=None, validate=False):
    """
    Instantiates model, defines which epoch to save as checkpoint, and trains
    """
    torch.set_float32_matmul_precision('medium')
    
    if not provide_model:
        if params.model == "sinr" or params.model == "log_reg":
            model = SINR(params, dataset)
        elif "sat" in params.model:
            model = SAT_SINR(params, dataset, get_embedder(params))
    else:
        model = provide_model
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath=params.local.cp_dir_path,
        filename=logger._name+"{val_loss:.4f}"
    )
    trainer = pl.Trainer(max_epochs=params.epochs, accelerator=("gpu" if params.local.gpu else "cpu"), devices=1,
                         precision="16-mixed", logger=logger, log_every_n_steps=50,
                         callbacks=[checkpoint_callback])
    if validate:
        trainer.validate(model=model, dataloaders=[val_loader])
    else:
        trainer.fit(model, train_loader, val_loader)
    return model


def load_cp(params, dataset):
    if params.model == "sinr" or params.model == "log_reg":
        model = SINR.load_from_checkpoint(params.checkpoint, params=params, dataset=dataset)
    elif "sat" in params.model:
        model = SAT_SINR.load_from_checkpoint(params.checkpoint, params=params, dataset=dataset, sent2_net=get_embedder(params))
    return model

        
@hydra.main(version_base=None, config_path='config', config_name='base_config.yaml')
def main(params):
    """main funct"""
    dataset, train_loader, val_loader = create_datasets(params)
    logger = get_logger(params, tag=params.tag)
    if params.checkpoint is not None:
        model = load_cp(params, dataset)
        model = train_model(params, dataset, train_loader, val_loader, provide_model=model, logger=logger, validate=params.validate)
    else:
        model = train_model(params, dataset, train_loader, val_loader, logger=logger)
    wandb.finish()

    
if __name__ == "__main__":
    main()
