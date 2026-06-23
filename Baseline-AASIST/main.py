"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import random
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (TrainDataset,TestDataset, genSpoof_list)
from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def enable_full_determinism() -> None:
    """Best-effort deterministic setup for CUDA/PyTorch."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # TF32 can slightly change numerical results across runs / devices.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some older builds / environments may not support this.
        pass


def _to_byte_tensor(x: Any) -> torch.ByteTensor:
    """Normalize RNG payloads to a CPU ByteTensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.uint8)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.astype(np.uint8, copy=False))
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.uint8)
    raise TypeError(f"Unsupported RNG state type: {type(x)}")


def capture_rng_state(train_generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """Capture all RNG states needed for epoch-boundary resume."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = [s.cpu() for s in torch.cuda.get_rng_state_all()]
    if train_generator is not None:
        state["train_generator"] = train_generator.get_state().cpu()
    return state


def restore_rng_state(state: Dict[str, Any],
                      train_generator: Optional[torch.Generator] = None) -> None:
    """Restore all RNG states saved by capture_rng_state()."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_to_byte_tensor(state["torch_cpu"]))
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all([_to_byte_tensor(s) for s in state["torch_cuda_all"]])
    if train_generator is not None and "train_generator" in state:
        train_generator.set_state(_to_byte_tensor(state["train_generator"]))


def move_optimizer_state_to_device(optimizer, device):
    """Move optimizer state tensors to the given device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def torch_load_compat(path, map_location=None):
    """torch.load wrapper that handles weights_only across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)
    enable_full_determinism()

    # define database related paths
    output_dir = Path(args.output_dir)
    trn_database_path = Path(config["trn_database_path"])
    trn_list_path = Path(config["trn_list_path"])
    dev_database_path = Path(config["dev_database_path"])
    dev_trial_path = Path(config["dev_trial_path"])
    # define model related paths
    model_tag = "{}_ep{}_bs{}".format(
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # Persistent generator for train DataLoader shuffle state.
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    # Define dataloaders.
    trn_loader, dev_loader = get_loader(
        trn_database_path, trn_list_path, dev_database_path, dev_trial_path,
        args.seed, config, train_generator
    )

    # evaluates pretrained model 
    # NOTE: Currently it is evaluated on the development set instead of the evaluation set
    if args.eval:
        model_path = args.eval_model_weights if args.eval_model_weights is not None else config["model_path"]
        model.load_state_dict(torch_load_compat(model_path, map_location=device))
        print("Model loaded : {}".format(model_path))
        print("Start evaluation...")
        produce_evaluation_file(dev_loader, model, device,
                                eval_score_path, dev_trial_path)

        eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=eval_score_path,
            output_file=model_tag/"loaded_model_result.txt")
        print("DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(eval_eer, eval_dcf, eval_cllr))

        """
        # Need asv score file for Track 2
        asv_score_path = ""
        eval_adcf, eval_tdcf, eval_teer = calculate_aDCF_tdcf_tEER(
            cm_scores_file=eval_score_path,
            asv_scores_file= asv_score_path,
            output_file=model_tag/"loaded_model_Phase2_result.txt")
        print("DONE. eval_adcf: {:.3f}, eval_tdcf:{:.5f} , eval_teer:{:.5f}".format(eval_adcf, eval_tdcf, eval_teer))
        """
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    spcen_params = []
    base_params = []
    for n, p in model.named_parameters():
        if 'spcen' in n:
            spcen_params.append(p)
        else:
            base_params.append(p)
    param_groups = [
        {"params": base_params},
        {"params": spcen_params, "lr": 1e-4}
    ]
    optimizer, scheduler = create_optimizer(param_groups, optim_config)
    optimizer_swa = SWA(optimizer)

    # Fix for torchcontrib SWA in PyTorch 2.0+
    for hook_name in [
        '_optimizer_step_pre_hooks', '_optimizer_step_post_hooks',
        '_optimizer_state_dict_pre_hooks', '_optimizer_state_dict_post_hooks',
        '_optimizer_load_state_dict_pre_hooks', '_optimizer_load_state_dict_post_hooks'
    ]:
        if not hasattr(optimizer_swa, hook_name):
            setattr(optimizer_swa, hook_name, {})
    if not hasattr(optimizer_swa, 'defaults'):
        optimizer_swa.defaults = optimizer.defaults

    best_dev_eer = 100.
    best_dev_dcf = 1.
    best_dev_cllr = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    start_epoch = 0

    if args.resume is not None:
        if os.path.exists(args.resume):
            print("Loading checkpoint from {}...".format(args.resume))
            checkpoint = torch_load_compat(args.resume, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                move_optimizer_state_to_device(optimizer, device)

                if "optimizer_swa_state_dict" in checkpoint:
                    optimizer_swa.load_state_dict(checkpoint["optimizer_swa_state_dict"])
                    move_optimizer_state_to_device(optimizer_swa, device)

                if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                start_epoch = checkpoint["epoch"] + 1
                best_dev_eer = checkpoint.get("best_dev_eer", 100.0)
                best_dev_dcf = checkpoint.get("best_dev_dcf", 1.0)
                best_dev_cllr = checkpoint.get("best_dev_cllr", 1.0)
                n_swa_update = checkpoint.get("n_swa_update", 0)

                if "rng_state" in checkpoint:
                    restore_rng_state(checkpoint["rng_state"], train_generator)

                print(f"Resumed training from epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights checkpoint. Starting from epoch 0.")
        else:
            print("Checkpoint path {} does not exist. Starting from scratch.".format(args.resume))

    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training loop.
    for epoch in range(start_epoch, config["num_epochs"]):
        print("training epoch{:03d}".format(epoch))
        
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_dcf, dev_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
            running_loss, dev_eer, dev_dcf, dev_cllr))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_dcf", dev_dcf, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)
        torch.save(model.state_dict(),
                       model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

        best_dev_dcf = min(dev_dcf, best_dev_dcf)
        best_dev_cllr = min(dev_cllr, best_dev_cllr)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            
            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_dcf, epoch)
        writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)

        # Save checkpoint for resume
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_swa_state_dict": optimizer_swa.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_dev_eer": best_dev_eer,
            "best_dev_dcf": best_dev_dcf,
            "best_dev_cllr": best_dev_cllr,
            "n_swa_update": n_swa_update,
            "rng_state": capture_rng_state(train_generator),
        }
        torch.save(checkpoint, model_save_path / "checkpoint_latest.pth")


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(trn_database_path: str, trn_list_path: str,
               dev_database_path: str, dev_trial_path: str,
               seed: int,
               config: dict,
               train_generator: torch.Generator) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / development."""

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = TrainDataset(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    trn_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = TestDataset(list_IDs=file_dev, base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    return trn_loader, dev_loader

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader, desc="Evaluating"):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    #assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            parts = trl.strip().split()
            spk_id = parts[0]
            utt_id = parts[1]
            key = parts[8] if len(parts) >= 10 else parts[5]
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(spk_id, utt_id, sco, key))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader, desc="Training"):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))

        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="path to checkpoint to resume training from")
    main(parser.parse_args())
