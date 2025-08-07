# standard libraries
import time
import os
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only
import wandb
from src.utils.data_utils.data_format_utils import to_time_aggregated_scores_torch
from src.utils.config_utils import add_all_attributes, add_attribute, generate_unique_name
from src.utils.data_utils.dataloading_utils import load_data, get_dataset_path, create_save_name
from src.utils.data_utils.data_format_utils import zero_out_diag_np
from src.utils.utils import write_results_to_disk
from src.utils.metrics_utils import evaluate_results
from src.model.generate_model import generate_model


@hydra.main(version_base=None, config_path="../configs/", config_name="main.yaml")
def run(cfg: DictConfig):
    if 'dataset' not in cfg:
        raise Exception("No dataset found in the config")

    dataset = cfg.dataset
    dataset_path = get_dataset_path(dataset)

    if dataset_path in cfg:
        model_config = cfg[dataset_path]
        print(dataset_path)
    else:
        raise Exception(
            "No model found in the config. Try running with option: python3 -m src.train +dataset=<dataset> +<dataset>=<model>")

    cfg.dataset_dir = os.path.join(cfg.dataset_dir, dataset_path)

    add_all_attributes(cfg, model_config)
    train(cfg)


def train(cfg):
    print("Running config:")
    print(cfg)
    dataset = cfg.dataset
    seed = int(cfg.random_seed)
    # set seed
    seed_everything(cfg.random_seed)

    X, adj_matrix, aggregated_graph, lag, data_dim = load_data(
        cfg.dataset, cfg.dataset_dir, cfg)

    add_attribute(cfg, 'lag', lag)
    add_attribute(cfg, 'aggregated_graph', aggregated_graph)
    add_attribute(cfg, 'num_nodes', X.shape[2])
    add_attribute(cfg, 'data_dim', data_dim)
    checkpoint_path='I:\\新建文件夹\\MCD-main\\logs\\mcd_netsim_15_200_seed_0_20241222_21_13_42\\version_0\\checkpoints\\epoch=36-step=4588.ckpt'
    # generate model
    model = generate_model(cfg)
    model = model(full_dataset=X,
                  adj_matrices=adj_matrix)
    # pass the dataset
    # model = model.load_from_checkpoint(checkpoint_path)

    model_name = cfg.model
    # log all hyperparameters
    model.eval()
    full_dataloader = model.get_full_dataloader()
    if 'use_indices' in cfg:
        f_path = os.path.join(cfg.dataset_dir, cfg.dataset,
                              f'{cfg.use_indices}_seed={cfg.random_seed}.npy')
        mix_idx = torch.Tensor(np.load(f_path))
        model.set_mixture_indices(mix_idx)

    training_needed = cfg.model in ['rhino', 'mcd']
    unique_name = generate_unique_name(cfg)
    csv_logger = CSVLogger("logs", name=unique_name)
    wandb_logger = WandbLogger(
        name=unique_name, project=cfg.wandb_project, log_model=True)

    # log all hyperparameters

    if rank_zero_only.rank == 0:
        for key in cfg:
            wandb_logger.experiment.config[key] = cfg[key]
    csv_logger.log_hyperparams(cfg)

    if training_needed:
        # either val_loss or likelihood
        monitor_checkpoint_based_on = cfg.monitor_checkpoint_based_on
        ckpt_choice = 'best'
        #用ModelCheckpoint回调保存最佳模型
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor=monitor_checkpoint_based_on, mode="min", save_last=True)

        # if len(cfg.gpu) > 1:
        #     strategy = 'ddp_find_unused_parameters_true'
        # else:
        #     strategy = 'auto'

        if 'val_every_n_epochs' in cfg:
            val_every_n_epochs = cfg.val_every_n_epochs
        else:
            val_every_n_epochs = 1

    trainer = Trainer(max_epochs=cfg.num_epochs,
                      accelerator="cpu",
                      # devices=cfg.cpu,
                      devices=1,
                      precision=cfg.precision,
                      logger=[csv_logger, wandb_logger],
                      callbacks=[checkpoint_callback],
                      # strategy=strategy,
                      enable_progress_bar=True,  # 进度条显示
                      check_val_every_n_epoch=val_every_n_epochs)
    trainer=trainer.load_from_checkpoint(checkpoint_path)
    L = trainer.predict(model, full_dataloader)

    #返回估计图[batchsize,lags,num_nodes,num_nodes]、估计的概率图（未二值化）[batchsize,lags,num_nodes,num_nodes]、
    # 真实图[num_nodes,num_nodes]、注意力分数矩阵[batchsize,lags,num_nodes,num_nodes]、所有潜在的K个图[K,lags,num_nodes,num_nodes]
    predictions = []
    scores = []
    adj_matrix = []
    attention=[]
    G=[]
    for graph, prob, matrix,att,g in L:
        predictions.append(graph)
        scores.append(prob)
        adj_matrix.append(matrix)
        attention.append(att)
        G.append(g)

    predictions = torch.concatenate(predictions, dim=0)
    scores = torch.concatenate(scores, dim=0)
    adj_matrix = torch.concatenate(adj_matrix, dim=0)
    attention=torch.concatenate(attention,dim=0)
    G = torch.mean(torch.stack(G), dim=0)

    # predictions=predictions.reshape(-1, 100,3, 5, 5).permute(1, 0, 2, 3,4)
    # adj_matrix = adj_matrix.reshape(-1, 100, 5, 5).permute(1, 0, 2, 3)
    # attention = attention.reshape(-1, 100, 3, 5, 5).permute(1, 0, 2, 3, 4)
    # scores = scores.reshape(-1, 100,3, 5, 5).permute(1, 0, 2, 3,4)
    #
    # predictions=predictions.reshape(-1, 115,3, 50, 50).permute(1, 0, 2, 3,4)
    # adj_matrix = adj_matrix.reshape(-1, 115, 50, 50).permute(1, 0, 2, 3)
    # attention = attention.reshape(-1, 115, 3, 50, 50).permute(1, 0, 2, 3, 4)
    # scores = scores.reshape(-1, 115,3, 50, 50).permute(1, 0, 2, 3,4)

    # predictions = predictions.reshape(-1, 5, 3, 10, 10).permute(1, 0, 2, 3, 4)
    # adj_matrix = adj_matrix.reshape(-1, 5, 10, 10).permute(1, 0, 2, 3)
    # attention = attention.reshape(-1, 5, 3, 10, 10).permute(1, 0, 2, 3, 4)
    # scores = scores.reshape(-1, 5, 3, 10, 10).permute(1, 0, 2, 3, 4)
    #将每个被试的样本排列到一起
    predictions = predictions.reshape(-1, 50, 3, 15, 15).permute(1, 0, 2, 3, 4)
    adj_matrix = adj_matrix.reshape(-1, 50, 15, 15).permute(1, 0, 2, 3)
    attention = attention.reshape(-1, 50, 3, 15, 15).permute(1, 0, 2, 3, 4)
    scores = scores.reshape(-1, 50, 3, 15, 15).permute(1, 0, 2, 3, 4)

    predictions=torch.mean(predictions,dim=1)#在划分的多个片段上进行平均
    adj_matrix=torch.mean(adj_matrix,dim=1)
    attention = torch.mean(attention, dim=1)
    scores = torch.mean(scores, dim=1)
    predictions = predictions.detach().cpu().numpy()
    print("pred:\n", predictions[0])

    scores = scores.detach().cpu().numpy()
    adj_matrix = adj_matrix.detach().cpu().numpy()
    attention = attention.detach().cpu().numpy()
    G= G.detach().cpu().numpy()



    if model_name == 'mcd':
        true_cluster_indices, pred_cluster_indices = model.get_cluster_indices()
    else:
        pred_cluster_indices = None
        true_cluster_indices = None
    predictions = np.max(predictions, axis=1)
    scores = np.max(scores, axis=1)
    if 'dream3' in dataset:
        predictions = zero_out_diag_np(predictions)
        scores = zero_out_diag_np(scores)

    metrics = evaluate_results(scores=scores,
                               adj_matrix=adj_matrix,
                               predictions=predictions,
                               aggregated_graph=aggregated_graph,
                               true_cluster_indices=true_cluster_indices,
                               pred_cluster_indices=pred_cluster_indices)
    print("metrics\n",metrics)

if __name__ == "__main__":
    run()
