# standard libraries
import time
import os
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only

from src.utils.data_utils.data_format_utils import to_time_aggregated_scores_torch
from src.utils.config_utils import add_all_attributes, add_attribute, generate_unique_name
from src.utils.data_utils.dataloading_utils import load_data, get_dataset_path, create_save_name
from src.utils.data_utils.data_format_utils import zero_out_diag_np
from src.utils.utils import write_results_to_disk
from src.utils.metrics_utils import evaluate_results
from src.model.generate_model import generate_model
import warnings

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)


@hydra.main(version_base=None, config_path="../configs/", config_name="main.yaml")
def run(cfg: DictConfig):
    if 'dataset' not in cfg:
        raise Exception("No dataset found in the config")

    dataset = cfg.dataset
    dataset_path = get_dataset_path(dataset)

    if dataset_path in cfg:
        model_config = cfg[dataset_path]
    else:
        raise Exception(
            "No model found in the config. Try running with option: python3 -m src.train +dataset=<dataset> +<dataset>=<model>")

    cfg.dataset_dir = os.path.join(cfg.dataset_dir, dataset_path)

    add_all_attributes(cfg, model_config)
    train(cfg)


def train(cfg):
    dataset = cfg.dataset
    seed = int(cfg.random_seed)
    # set random seed
    seed_everything(cfg.random_seed, workers=True)
    X, adj_matrix, aggregated_graph, lag, data_dim = load_data(
        cfg.dataset, cfg.dataset_dir, cfg)

    add_attribute(cfg, 'lag', lag)
    add_attribute(cfg, 'aggregated_graph', aggregated_graph)
    add_attribute(cfg, 'num_nodes', X.shape[2])
    add_attribute(cfg, 'data_dim', data_dim)

    # model
    model = generate_model(cfg)

    # dataset
    model = model(full_dataset=X,
                  adj_matrices=adj_matrix)
    model_name = cfg.model

    if 'use_indices' in cfg:
        f_path = os.path.join(cfg.dataset_dir, cfg.dataset,
                              f'{cfg.use_indices}_seed={cfg.random_seed}.npy')
        mix_idx = torch.Tensor(np.load(f_path))
        model.set_mixture_indices(mix_idx)

    training_needed = cfg.model in ['rhino', 'mcd', 'tcmm']
    unique_name = generate_unique_name(cfg)
    csv_logger = CSVLogger("logs", name=unique_name)

    csv_logger.log_hyperparams(cfg)

    if training_needed:
        # either val_loss or likelihood
        checkpoint_path = cfg.get('checkpoint_path', None)
        monitor_checkpoint_based_on = cfg.monitor_checkpoint_based_on
        ckpt_choice = 'best'
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1, monitor=monitor_checkpoint_based_on, mode="min", save_last=True)

        if len(cfg.gpu) > 1:
             strategy = 'ddp_find_unused_parameters_true'
        else:
             strategy = 'auto'

        if 'val_every_n_epochs' in cfg:
            val_every_n_epochs = cfg.val_every_n_epochs
        else:
            val_every_n_epochs = 1

        trainer = Trainer(max_epochs=cfg.num_epochs,
                          accelerator="gpu",
                          devices=cfg.gpu,
                          precision=cfg.precision,
                          logger=csv_logger,
                          callbacks=[checkpoint_callback],
                          strategy=strategy,
                          enable_progress_bar=True,
                          check_val_every_n_epoch=val_every_n_epochs,
                          )

        summary = ModelSummary(model, max_depth=10)
        print(summary)

        start_time = time.time()
        if checkpoint_path is None:
            trainer.fit(model=model)
        else:
            trainer.fit(model=model, ckpt_path=checkpoint_path)
        end_time = time.time()

        print("Model trained in", str(end_time - start_time) + "s")
    else:
        if cfg.gpu != -1:
            print("WARNING: GPU specified, but baseline cannot use GPU.")
        trainer = Trainer(logger=csv_logger,
                          accelerator='cpu')

    full_dataloader = model.get_full_dataloader()
    if training_needed:
        model.eval()
        L = trainer.predict(model, full_dataloader, ckpt_path=ckpt_choice)
    else:
        L = trainer.predict(model, full_dataloader)

    predictions = []
    scores = []
    adj_matrix = []
    for graph, prob, matrix in L:
        predictions.append(graph)
        scores.append(prob)
        adj_matrix.append(matrix)

    predictions = torch.concatenate(predictions, dim=0)
    scores = torch.concatenate(scores, dim=0)
    adj_matrix = torch.concatenate(adj_matrix, dim=0)

    predictions = predictions.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    adj_matrix = adj_matrix.detach().cpu().numpy()

    if dataset in ['ecoli1', 'ecoli2', 'yeast1', 'yeast2', 'yeast3',
                   'dream3_combined']:  # ignore self-connections in dream3 dataset
        predictions = zero_out_diag_np(predictions)
        scores = zero_out_diag_np(scores)

    # save predictions
    if not os.path.exists(os.path.join('results', dataset)):
        os.makedirs(os.path.join('results', dataset))

    if training_needed and ckpt_choice == 'best':
        if model_name in ('mcd', 'tcmm'):
            np.save(os.path.join('results', dataset,
                                 f'{model_name}_{checkpoint_callback.best_model_score.item()}_k{cfg.trainer.num_graphs}.npy'),
                    scores)
        else:
            np.save(os.path.join('results', dataset,
                                 f'{model_name}_{checkpoint_callback.best_model_score.item()}.npy'), scores)
    else:
        np.save(os.path.join('results', dataset, f'{model_name}.npy'), scores)

    metrics = evaluate_results(scores=scores,
                               adj_matrix=adj_matrix,
                               predictions=predictions,
                               aggregated_graph=aggregated_graph)

    # add the dataset name and model to the csv
    metrics['model'] = model_name + "_seed_" + str(seed)

    if training_needed and ckpt_choice == 'best':
        metrics['best_loss'] = checkpoint_callback.best_model_score.item()
    metrics['dataset'] = dataset

    # write the results to the log directory
    write_results_to_disk(create_save_name(dataset, cfg), metrics)


if __name__ == "__main__":
    run()