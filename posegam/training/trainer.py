# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2025 WindVChen.
# All rights reserved.
#
# This source code is derived from VGGT and licensed under the VGGT License
# found in the LICENSE_VGGT file in the root directory of this source tree.

import contextlib
import gc
import json
import logging
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from posegam.training.train_utils.checkpoint import DDPCheckpointSaver
from posegam.training.train_utils.distributed import get_machine_local_and_dist_rank
from posegam.training.train_utils.freeze import freeze_modules
from posegam.training.train_utils.general import *
from posegam.training.train_utils.logging import setup_logging
from posegam.training.train_utils.optimizer import construct_optimizers

from posegam.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri
from posegam.utils.rotation import mat_to_quat
from posegam.utils.geometry import closed_form_inverse_se3

import yaml
from omegaconf import OmegaConf

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        
        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0
        
        # AUC metrics accumulation storage
        self.auc_accumulator = {'train': {}, 'val': {}}

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        if self.rank == 0:
            # save all the args
            cfg_path = os.path.join(self.logging_conf.log_dir, "cfg.yaml")
            with open(cfg_path, "w") as f:                
                # Convert OmegaConf objects to regular dictionaries for serialization
                def convert_config(obj):
                    if hasattr(obj, '_content') or hasattr(obj, '_metadata'):  # OmegaConf object
                        return OmegaConf.to_container(obj, resolve=True)
                    return obj
                
                config_dict = {
                    "data": convert_config(self.data_conf),
                    "model": convert_config(self.model_conf),
                    "loss": convert_config(self.loss_conf),
                    "logging": convert_config(self.logging_conf),
                    "checkpoint": convert_config(self.checkpoint_conf),
                    "optim": convert_config(self.optim_conf),
                    "max_epochs": self.max_epochs,
                    "mode": self.mode,
                    "device": device,
                    "seed_value": self.seed_value,
                    "val_epoch_freq": self.val_epoch_freq,
                    "distributed": convert_config(distributed),
                    "cuda": convert_config(cuda),
                    "limit_train_batches": self.limit_train_batches,
                    "limit_val_batches": self.limit_val_batches,
                    "env_variables": convert_config(env_variables),
                    "accum_steps": self.accum_steps,
                    **{k: convert_config(v) for k, v in kwargs.items()},  # Convert kwargs too
                }
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            # Use sys.modules to access the actual logging module instead of the parameter
            log_module = sys.modules['logging']
            log_module.info(f"Config saved to {cfg_path}")

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Resume from an explicit checkpoint, else auto-resume from save_dir, else (training
        # from scratch) initialize the matching layers from the pretrained VGGT-1B backbone.
        # An empty resume_checkpoint_path means "no explicit checkpoint".
        if self.checkpoint_conf.resume_checkpoint_path:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)
            else:
                self._load_vggt_pretrained()

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)
        
        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_vggt_pretrained(self):
        """Initialize matching layers from the pretrained VGGT-1B backbone.

        Used only when training from scratch (i.e. not resuming from a checkpoint). Only the
        parameters whose name AND shape match are copied; PoseGAM-specific layers and the
        SONATA encoder keep their existing initialization (loaded with strict=False).
        """
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        logging.info(f"Initializing matching layers from VGGT-1B ({_URL}) (rank {self.rank})")

        # Download once on rank 0 (populating the torch.hub cache); other ranks wait at the
        # barrier and then load the cached file, avoiding a concurrent-download race.
        state_dict_vggt = None
        if self.rank == 0:
            state_dict_vggt = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        if is_dist_avail_and_initialized():
            dist.barrier()
        if state_dict_vggt is None:
            state_dict_vggt = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")

        model_state_dict = self.model.state_dict()

        # Keep only keys present in the model with a matching shape.
        filtered_state_dict = {}
        for k in list(state_dict_vggt.keys()):
            v = state_dict_vggt[k]
            if k in model_state_dict and v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            # Free memory as we go.
            if k not in filtered_state_dict:
                state_dict_vggt.pop(k, None)

        self.model.load_state_dict(filtered_state_dict, strict=False)

        # Report exactly which tensors were initialized from VGGT-1B (rank 0 only).
        if self.rank == 0:
            loaded_keys = sorted(filtered_state_dict.keys())
            logging.info(
                f"Loaded {len(loaded_keys)} tensors from VGGT-1B into the model:\n"
                + "\n".join(f"  {k}" for k in loaded_keys)
            )

        # Free the checkpoint tensors promptly.
        del state_dict_vggt, filtered_state_dict, model_state_dict
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        
        # Extract model state dict without duplicating memory and prune in-place
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            # Pop to free memory from the root checkpoint dict early
            checkpoint_state_dict = checkpoint.pop("model")
        else:
            # Older checkpoints might be just the state_dict
            checkpoint_state_dict = checkpoint

        # Build a lightweight mapping of expected tensor shapes from the current model
        param_shapes = {}
        for name, p in self.model.named_parameters(recurse=True):
            param_shapes[name] = tuple(p.shape)
        for name, b in self.model.named_buffers(recurse=True):
            param_shapes[name] = tuple(b.shape)

        # Prune keys directly from the loaded state_dict to avoid creating a second dict copy
        kept_keys = 0
        for k in list(checkpoint_state_dict.keys()):
            v = checkpoint_state_dict[k]
            if k not in param_shapes:
                if self.rank == 0:
                    logging.warning(f"Skipping {k}: not found in current model")
                checkpoint_state_dict.pop(k)
                continue
            if tuple(v.shape) != param_shapes[k]:
                if self.rank == 0:
                    logging.warning(
                        f"Skipping {k}: shape mismatch {tuple(v.shape)} vs {param_shapes[k]}"
                    )
                checkpoint_state_dict.pop(k)
                continue
            kept_keys += 1

        # Load directly from the pruned dict
        missing, unexpected = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")
            logging.info(f"Successfully loaded {kept_keys} weights from checkpoint.")

        # Clear state dict to free memory ASAP (keep the main checkpoint for metadata/optimizer)
        if checkpoint_state_dict is not checkpoint:
            checkpoint_state_dict.clear()

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint and self.mode != "val":
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            # Pop to reduce peak memory during load
            optimizer_state = checkpoint.pop("optimizer")
            if isinstance(optimizer_state, list):
                # Multiple optimizers case
                for i, optim in enumerate(self.optims):
                    optim.optimizer.load_state_dict(optimizer_state[i])
            else:
                # Single optimizer case
                self.optims[0].optimizer.load_state_dict(optimizer_state)
            # Clean up optimizer state after loading
            del optimizer_state

        # Load training progress
        if "prev_epoch" in checkpoint:
            self.epoch = checkpoint.get("prev_epoch", 0)
        elif "epoch" in checkpoint:
            # Fallback for older checkpoint format
            self.epoch = checkpoint.get("epoch", 0)
        self.steps = checkpoint.get("steps", {"train": 0, "val": 0})
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            scaler_state = checkpoint.pop("scaler")
            self.scaler.load_state_dict(scaler_state)
            del scaler_state

        # Clean up checkpoint data and force garbage collection
        # If we reached here with an old-format checkpoint where `checkpoint` is actually the state_dict,
        # ensure we drop all references.
        if isinstance(checkpoint, dict):
            checkpoint.clear()
        del checkpoint
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }
        
        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models = None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )




    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def _should_calculate_auc_metrics(self, phase: str) -> bool:
        """Determines if AUC metrics should be calculated for the current epoch."""
        if phase == 'val':
            # Always calculate AUC for validation
            return True
        elif phase == 'train':
            # Only calculate AUC for training during saving epochs
            should_calc = (
                int(self.epoch) % 10 == 0
                and (int(self.epoch) > 0)
            )
            if should_calc and self.rank == 0:
                logging.info(f"Will calculate AUC metrics for training epoch {self.epoch} (saving epoch)")
            return should_calc
        return False

    def _reset_auc_accumulator(self, phase: str):
        """Reset AUC accumulator for a new epoch."""
        self.auc_accumulator[phase] = {
            'rotation_errors': [],
            'translation_errors': [],
            'valid_samples': 0,
            # Separate accumulators for masked and unmasked cameras
            'masked_rotation_errors': [],
            'masked_translation_errors': [],
            'masked_valid_samples': 0,
            'unmasked_rotation_errors': [],
            'unmasked_translation_errors': [],
            'unmasked_valid_samples': 0,
            # Track error type for logging purposes
            'error_type': None,  # Will be set to 'absolute' or 'relative'
        }

    def _accumulate_auc_metrics(self, phase: str, rotation_errors: torch.Tensor, translation_errors: torch.Tensor, 
                               masked_rotation_errors: torch.Tensor = None, masked_translation_errors: torch.Tensor = None,
                               unmasked_rotation_errors: torch.Tensor = None, unmasked_translation_errors: torch.Tensor = None,
                               error_type: str = 'relative'):
        """Accumulate AUC metrics for the current batch."""
        if phase not in self.auc_accumulator:
            self._reset_auc_accumulator(phase)
        
        # Set error type on first accumulation
        if self.auc_accumulator[phase]['error_type'] is None:
            self.auc_accumulator[phase]['error_type'] = error_type
        
        # Move tensors to CPU and convert to numpy for storage efficiency
        rot_errors_np = rotation_errors.cpu().numpy()
        trans_errors_np = translation_errors.cpu().numpy()
        
        self.auc_accumulator[phase]['rotation_errors'].extend(rot_errors_np.tolist())
        self.auc_accumulator[phase]['translation_errors'].extend(trans_errors_np.tolist())
        self.auc_accumulator[phase]['valid_samples'] += len(rotation_errors)
        
        # Accumulate masked camera errors if provided
        if masked_rotation_errors is not None and len(masked_rotation_errors) > 0:
            masked_rot_errors_np = masked_rotation_errors.cpu().numpy()
            masked_trans_errors_np = masked_translation_errors.cpu().numpy()
            self.auc_accumulator[phase]['masked_rotation_errors'].extend(masked_rot_errors_np.tolist())
            self.auc_accumulator[phase]['masked_translation_errors'].extend(masked_trans_errors_np.tolist())
            self.auc_accumulator[phase]['masked_valid_samples'] += len(masked_rotation_errors)
        
        # Accumulate unmasked camera errors if provided
        if unmasked_rotation_errors is not None and len(unmasked_rotation_errors) > 0:
            unmasked_rot_errors_np = unmasked_rotation_errors.cpu().numpy()
            unmasked_trans_errors_np = unmasked_translation_errors.cpu().numpy()
            self.auc_accumulator[phase]['unmasked_rotation_errors'].extend(unmasked_rot_errors_np.tolist())
            self.auc_accumulator[phase]['unmasked_translation_errors'].extend(unmasked_trans_errors_np.tolist())
            self.auc_accumulator[phase]['unmasked_valid_samples'] += len(unmasked_rotation_errors)
        
        # Debug logging (only occasionally to avoid spam)
        if self.auc_accumulator[phase]['valid_samples'] % 100 == 0 and self.rank == 0:
            logging.info(f"Accumulated {self.auc_accumulator[phase]['valid_samples']} AUC samples for {phase} "
                        f"(masked: {self.auc_accumulator[phase]['masked_valid_samples']}, "
                        f"unmasked: {self.auc_accumulator[phase]['unmasked_valid_samples']}) (rank {self.rank})")

    def _compute_epoch_auc_metrics(self, phase: str) -> Dict[str, float]:
        """Compute AUC metrics for the entire epoch."""
        if phase not in self.auc_accumulator or self.auc_accumulator[phase]['valid_samples'] == 0:
            return {}
        
        auc_metrics = {}
        
        # Compute overall AUC metrics (all camera pairs)
        rotation_errors = torch.tensor(self.auc_accumulator[phase]['rotation_errors'], device=self.device)
        translation_errors = torch.tensor(self.auc_accumulator[phase]['translation_errors'], device=self.device)
        
        auc_metrics['overall_auc_3'] = self.calculate_auc_torch(rotation_errors, translation_errors, max_threshold=3)
        auc_metrics['overall_auc_5'] = self.calculate_auc_torch(rotation_errors, translation_errors, max_threshold=5)
        auc_metrics['overall_auc_10'] = self.calculate_auc_torch(rotation_errors, translation_errors, max_threshold=10)
        auc_metrics['overall_auc_30'] = self.calculate_auc_torch(rotation_errors, translation_errors, max_threshold=30)

        # Compute masked camera AUC metrics
        if self.auc_accumulator[phase]['masked_valid_samples'] > 0:
            masked_rotation_errors = torch.tensor(self.auc_accumulator[phase]['masked_rotation_errors'], device=self.device)
            masked_translation_errors = torch.tensor(self.auc_accumulator[phase]['masked_translation_errors'], device=self.device)
            
            auc_metrics['known_cam_auc_3'] = self.calculate_auc_torch(masked_rotation_errors, masked_translation_errors, max_threshold=3)
            auc_metrics['known_cam_auc_5'] = self.calculate_auc_torch(masked_rotation_errors, masked_translation_errors, max_threshold=5)
            auc_metrics['known_cam_auc_10'] = self.calculate_auc_torch(masked_rotation_errors, masked_translation_errors, max_threshold=10)
            auc_metrics['known_cam_auc_30'] = self.calculate_auc_torch(masked_rotation_errors, masked_translation_errors, max_threshold=30)

        # Compute unmasked camera AUC metrics
        if self.auc_accumulator[phase]['unmasked_valid_samples'] > 0:
            unmasked_rotation_errors = torch.tensor(self.auc_accumulator[phase]['unmasked_rotation_errors'], device=self.device)
            unmasked_translation_errors = torch.tensor(self.auc_accumulator[phase]['unmasked_translation_errors'], device=self.device)
            
            auc_metrics['unknown_cam_auc_3'] = self.calculate_auc_torch(unmasked_rotation_errors, unmasked_translation_errors, max_threshold=3)
            auc_metrics['unknown_cam_auc_5'] = self.calculate_auc_torch(unmasked_rotation_errors, unmasked_translation_errors, max_threshold=5)
            auc_metrics['unknown_cam_auc_10'] = self.calculate_auc_torch(unmasked_rotation_errors, unmasked_translation_errors, max_threshold=10)
            auc_metrics['unknown_cam_auc_30'] = self.calculate_auc_torch(unmasked_rotation_errors, unmasked_translation_errors, max_threshold=30)

        # Convert to float for logging
        for key in auc_metrics:
            auc_metrics[key] = auc_metrics[key].item()
            
        if self.rank == 0:
            error_type = self.auc_accumulator[phase].get('error_type', 'relative')
            error_desc = f"{error_type} pose"
            count_desc = 'poses' if error_type == 'absolute' else 'pairs'
            
            logging.info(f"Computed AUC metrics for {phase} epoch {self.epoch} using {error_desc} errors: "
                        f"Overall - AUC@3°={auc_metrics.get('overall_auc_3', 0):.4f}, "
                        f"AUC@5°={auc_metrics.get('overall_auc_5', 0):.4f}, "
                        f"AUC@10°={auc_metrics.get('overall_auc_10', 0):.4f}, "
                        f"AUC@30°={auc_metrics.get('overall_auc_30', 0):.4f} "
                        f"(from {len(rotation_errors)} {count_desc})")
            
            if self.auc_accumulator[phase]['masked_valid_samples'] > 0:
                logging.info(f"Masked cameras - AUC@3°={auc_metrics.get('known_cam_auc_3', 0):.4f}, "
                            f"AUC@5°={auc_metrics.get('known_cam_auc_5', 0):.4f}, "
                            f"AUC@10°={auc_metrics.get('known_cam_auc_10', 0):.4f}, "
                            f"AUC@30°={auc_metrics.get('known_cam_auc_30', 0):.4f} "
                            f"(from {self.auc_accumulator[phase]['masked_valid_samples']} {count_desc})")
            
            if self.auc_accumulator[phase]['unmasked_valid_samples'] > 0:
                logging.info(f"Unmasked cameras - AUC@3°={auc_metrics.get('unknown_cam_auc_3', 0):.4f}, "
                            f"AUC@5°={auc_metrics.get('unknown_cam_auc_5', 0):.4f}, "
                            f"AUC@10°={auc_metrics.get('unknown_cam_auc_10', 0):.4f}, "
                            f"AUC@30°={auc_metrics.get('unknown_cam_auc_30', 0):.4f} "
                            f"(from {self.auc_accumulator[phase]['unmasked_valid_samples']} {count_desc})")
                
        return auc_metrics

    def _log_epoch_auc_metrics(self, phase: str, auc_metrics: Dict[str, float]):
        """Log AUC metrics for the epoch."""
        if not auc_metrics or self.rank != 0:
            return
            
        step = self.steps[phase]
        
        # Log overall AUC metrics
        for metric_name in ['overall_auc_3', 'overall_auc_5', 'overall_auc_10', 'overall_auc_30']:
            if metric_name in auc_metrics:
                self.tb_writer.log(f"AUC/{phase}/{metric_name}", auc_metrics[metric_name], step)
                logging.info(f"Epoch {self.epoch} {phase} {metric_name}: {auc_metrics[metric_name]:.4f}")
        
        # Log known camera AUC metrics
        for metric_name in ['known_cam_auc_3', 'known_cam_auc_5', 'known_cam_auc_10', 'known_cam_auc_30']:
            if metric_name in auc_metrics:
                self.tb_writer.log(f"AUC/{phase}/{metric_name}", auc_metrics[metric_name], step)
                logging.info(f"Epoch {self.epoch} {phase} {metric_name}: {auc_metrics[metric_name]:.4f}")

        # Log unknown camera AUC metrics
        for metric_name in ['unknown_cam_auc_3', 'unknown_cam_auc_5', 'unknown_cam_auc_10', 'unknown_cam_auc_30']:
            if metric_name in auc_metrics:
                self.tb_writer.log(f"AUC/{phase}/{metric_name}", auc_metrics[metric_name], step)
                logging.info(f"Epoch {self.epoch} {phase} {metric_name}: {auc_metrics[metric_name]:.4f}")
        
        # Flush TensorBoard writer
        if hasattr(self.tb_writer, '_writer') and self.tb_writer._writer:
            self.tb_writer._writer.flush()

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)
            
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)
            
            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()
            
            self.epoch += 1
        
        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)
        
        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'
        
        # Reset AUC accumulator for validation
        self._reset_auc_accumulator(phase)
        if self.rank == 0:
            logging.info(f"Initialized AUC accumulator for {phase} epoch {self.epoch}")
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16
            
            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        # Compute and log epoch-level AUC metrics for validation
        with torch.no_grad():
            auc_metrics = self._compute_epoch_auc_metrics(phase)
            self._log_epoch_auc_metrics(phase, auc_metrics)

        return True

    def train_epoch(self, train_loader):        
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'
        
        # Reset AUC accumulator if we're calculating AUC metrics this epoch
        if self._should_calculate_auc_metrics(phase):
            self._reset_auc_accumulator(phase)
            if self.rank == 0:
                logging.info(f"Initialized AUC accumulator for {phase} epoch {self.epoch}")
        
        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }
        
        for config in self.gradient_clipper.configs: 
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")


        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter > limit_train_batches:
                break
            
            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)
            
            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps==1 or batch['images'].shape[0] == 1:
                # If accum_steps is 1 or batch size is 1, no need to chunk
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters
            )

            # After loss.backward()
            if data_iter == 0 and self.rank == 0 and self.epoch == 0:
                print("Checking for unused parameters:")
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is None:
                        print(f"UNUSED PARAMETER: {name}")

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            
            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )
                    
            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:   
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        # Compute and log epoch-level AUC metrics if calculated this epoch
        if self._should_calculate_auc_metrics(phase):
            with torch.no_grad():
                auc_metrics = self._compute_epoch_auc_metrics(phase)
                self._log_epoch_auc_metrics(phase, auc_metrics)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """        
        
        for optim in self.optims:   
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16
        
        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(
                        chunked_batch, self.model, phase, loss_meters
                    )


                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)


    def _process_batch(self, batch: Mapping):
        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.
        
        Returns:
            A dictionary containing the computed losses.
        """

        """Forward pass through the model."""

        images = batch['images']
        base_colors = batch['base_colors']
        point_sonata = batch['point_sonata']
        initial_sonata_num = batch['initial_sonata_num']

        world_points = batch['world_points'].permute(0, 1, 4, 2, 3)
        point_masks = batch['point_masks'].unsqueeze(-1).permute(0, 1, 4, 2, 3)

        camera_params = extri_intri_to_pose_encoding(batch['extrinsics'],
                                                     batch['intrinsics'], images[0, 0].shape[-2:])

        # Run model
        y_hat = model(images, camera_params, batch['camera_mask'], world_points, point_masks, base_colors, point_sonata, initial_sonata_num)

        # Loss computation
        loss_dict = self.loss(y_hat, batch)

        # Conditionally evaluate and accumulate AUC metrics
        auc_metrics = {}
        if self._should_calculate_auc_metrics(phase):
            with torch.no_grad():
                auc_metrics = self._evaluate_and_accumulate_auc_metrics(y_hat, batch, phase)
        
        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch, **auc_metrics}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data['extrinsics'].shape[0]
        
        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)
        
        # Ensure TensorBoard writer is flushed for real-time monitoring
        if step % self.logging_conf.log_freq == 0 and self.rank == 0:
            if hasattr(self.tb_writer, '_writer') and self.tb_writer._writer:
                self.tb_writer._writer.flush()

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
            self.logging_conf.log_visuals
            and (phase in self.logging_conf.log_visual_frequency)
            and self.logging_conf.log_visual_frequency[phase] > 0
            and (step % self.logging_conf.log_visual_frequency[phase] == 0)
            and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )

    def build_pair_index(self, N, B=1):
        """
        Build indices for all possible pairs of frames.

        Args:
            N: Number of frames
            B: Batch size

        Returns:
            i1, i2: Indices for all possible pairs
        """
        i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
        i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
        return i1, i2

    def build_masked_unmasked_pair_indices(self, N, camera_mask):
        """
        Build indices for camera pairs separated by mask status.

        Args:
            N: Number of frames
            camera_mask: Tensor of shape [N] where 1 = masked, 0 = unmasked

        Returns:
            masked_pairs: (i1, i2) indices for pairs involving at least one masked camera
            unmasked_pairs: (i1, i2) indices for pairs involving only unmasked cameras
        """
        # Get all possible pairs
        i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
        
        # Determine mask status for each camera in pairs
        mask1 = camera_mask[i1_]  # Mask status of first camera in each pair
        mask2 = camera_mask[i2_]  # Mask status of second camera in each pair
        
        # Pairs involving only masked camera (both cameras are masked)
        masked_pair_mask = (mask1 == 1) & (mask2 == 1)
        masked_pair_mask = masked_pair_mask.cpu()
        masked_i1 = i1_[masked_pair_mask]
        masked_i2 = i2_[masked_pair_mask]
        
        # Pairs involving only unmasked cameras (both cameras are unmasked)
        unmasked_pair_mask = (mask1 == 0) & (mask2 == 0)
        unmasked_pair_mask = unmasked_pair_mask.cpu()
        unmasked_i1 = i1_[unmasked_pair_mask]
        unmasked_i2 = i2_[unmasked_pair_mask]
        
        return (masked_i1, masked_i2), (unmasked_i1, unmasked_i2)
    
    def rotation_angle(self, rot_gt, rot_pred, batch_size=None, eps=1e-15):
        """
        Calculate rotation angle error between ground truth and predicted rotations.

        Args:
            rot_gt: Ground truth rotation matrices
            rot_pred: Predicted rotation matrices
            batch_size: Batch size for reshaping the result
            eps: Small value to avoid numerical issues

        Returns:
            Rotation angle error in degrees
        """
        q_pred = mat_to_quat(rot_pred)
        q_gt = mat_to_quat(rot_gt)

        loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
        err_q = torch.arccos(1 - 2 * loss_q)

        rel_rangle_deg = err_q * 180 / np.pi

        if batch_size is not None:
            rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

        return rel_rangle_deg

    def translation_angle(self, tvec_gt, tvec_pred, batch_size=None, ambiguity=False):
        """
        Calculate translation angle error between ground truth and predicted translations.

        Args:
            tvec_gt: Ground truth translation vectors
            tvec_pred: Predicted translation vectors
            batch_size: Batch size for reshaping the result
            ambiguity: Whether to handle direction ambiguity

        Returns:
            Translation angle error in degrees
        """
        rel_tangle_deg = self.compare_translation_by_angle(tvec_gt, tvec_pred)
        rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

        if ambiguity:
            rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

        if batch_size is not None:
            rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

        return rel_tangle_deg

    def compare_translation_by_angle(self, t_gt, t, eps=1e-15, default_err=1e6):
        """
        Normalize the translation vectors and compute the angle between them.

        Args:
            t_gt: Ground truth translation vectors
            t: Predicted translation vectors
            eps: Small value to avoid division by zero
            default_err: Default error value for invalid cases

        Returns:
            Angular error between translation vectors in radians
        """
        t_norm = torch.norm(t, dim=1, keepdim=True)
        t = t / (t_norm + eps)

        t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
        t_gt = t_gt / (t_gt_norm + eps)

        loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
        err_t = torch.acos(torch.sqrt(1 - loss_t))

        err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
        return err_t

    def calculate_auc_torch(self, r_error, t_error, max_threshold=30):
        """
        Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

        Args:
            r_error: torch tensor representing R error values (Degree)
            t_error: torch tensor representing T error values (Degree)
            max_threshold: Maximum threshold value for binning the histogram

        Returns:
            AUC value as torch tensor
        """
        error_matrix = torch.stack((r_error, t_error), dim=1)
        max_errors = torch.max(error_matrix, dim=1)[0]
        
        # Clamp max_errors to be within [0, max_threshold]
        max_errors = torch.clamp(max_errors, 0, max_threshold + 1e-6)
        
        # Create histogram using bincount
        bins = torch.arange(max_threshold + 1, device=r_error.device, dtype=torch.float32)
        digitized = torch.bucketize(max_errors, bins, right=True) # There will be one off position.
        histogram = torch.bincount(digitized, minlength=max_threshold + 1)[1:max_threshold+1]
        
        num_pairs = float(len(max_errors))
        normalized_histogram = histogram.float() / num_pairs
        return torch.mean(torch.cumsum(normalized_histogram, dim=0))

    def se3_to_relative_pose_error(self, pred_se3, gt_se3, num_frames, camera_mask):
        """
        Compute rotation and translation errors between predicted and ground truth poses.

        Args:
            pred_se3: Predicted SE(3) transformations
            gt_se3: Ground truth SE(3) transformations
            num_frames: Number of frames
            camera_mask: Optional mask tensor with shape [num_frames] where 1 = masked, 0 = unmasked

        Returns:
            If camera_mask is None:
                Rotation and translation angle errors in degrees
            If camera_mask is provided:
                (overall_errors, masked_errors, unmasked_errors) where each is (rot_errors, trans_errors)
        """
        # Separate pairs by mask status
        (masked_i1, masked_i2), (unmasked_i1, unmasked_i2) = self.build_masked_unmasked_pair_indices(num_frames, camera_mask)
        
        # Get all pairs for overall computation
        pair_idx_i1, pair_idx_i2 = self.build_pair_index(num_frames)
        
        # Compute overall relative pose errors
        relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
            gt_se3[pair_idx_i2]
        )
        relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
            pred_se3[pair_idx_i2]
        )
        
        overall_rel_rangle_deg = self.rotation_angle(
            relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
        )
        overall_rel_tangle_deg = self.translation_angle(
            relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
        )
        
        # Compute masked pairs errors
        masked_rel_rangle_deg = torch.tensor([], device=pred_se3.device)
        masked_rel_tangle_deg = torch.tensor([], device=pred_se3.device)
        if len(masked_i1) > 0:
            masked_relative_pose_gt = closed_form_inverse_se3(gt_se3[masked_i1]).bmm(
                gt_se3[masked_i2]
            )
            masked_relative_pose_pred = closed_form_inverse_se3(pred_se3[masked_i1]).bmm(
                pred_se3[masked_i2]
            )
            
            masked_rel_rangle_deg = self.rotation_angle(
                masked_relative_pose_gt[:, :3, :3], masked_relative_pose_pred[:, :3, :3]
            )
            masked_rel_tangle_deg = self.translation_angle(
                masked_relative_pose_gt[:, :3, 3], masked_relative_pose_pred[:, :3, 3]
            )
        
        # Compute unmasked pairs errors
        unmasked_rel_rangle_deg = torch.tensor([], device=pred_se3.device)
        unmasked_rel_tangle_deg = torch.tensor([], device=pred_se3.device)
        if len(unmasked_i1) > 0:
            unmasked_relative_pose_gt = closed_form_inverse_se3(gt_se3[unmasked_i1]).bmm(
                gt_se3[unmasked_i2]
            )
            unmasked_relative_pose_pred = closed_form_inverse_se3(pred_se3[unmasked_i1]).bmm(
                pred_se3[unmasked_i2]
            )
            
            unmasked_rel_rangle_deg = self.rotation_angle(
                unmasked_relative_pose_gt[:, :3, :3], unmasked_relative_pose_pred[:, :3, :3]
            )
            unmasked_rel_tangle_deg = self.translation_angle(
                unmasked_relative_pose_gt[:, :3, 3], unmasked_relative_pose_pred[:, :3, 3]
            )
        
        return ((overall_rel_rangle_deg, overall_rel_tangle_deg),
                (masked_rel_rangle_deg, masked_rel_tangle_deg),
                (unmasked_rel_rangle_deg, unmasked_rel_tangle_deg))


    def se3_to_absolute_pose_error(self, pred_se3, gt_se3, camera_mask):
        """
        Compute absolute rotation and translation errors between predicted and ground truth poses.
        This is used when num_frames = 1 (single frame case).

        Args:
            pred_se3: Predicted SE(3) transformations with shape [N, 4, 4]
            gt_se3: Ground truth SE(3) transformations with shape [N, 4, 4] 
            camera_mask: Mask tensor with shape [N] where 1 = masked, 0 = unmasked

        Returns:
            (overall_errors, masked_errors, unmasked_errors) where each is (rot_errors, trans_errors)
        """
        num_frames = pred_se3.shape[0]
        
        # Compute overall absolute pose errors
        overall_abs_rangle_deg = self.rotation_angle(
            gt_se3[:, :3, :3], pred_se3[:, :3, :3]
        )
        overall_abs_tangle_deg = self.translation_angle(
            gt_se3[:, :3, 3], pred_se3[:, :3, 3]
        )
        
        # Compute masked camera errors
        masked_indices = (camera_mask == 1).nonzero(as_tuple=True)[0]
        masked_abs_rangle_deg = torch.tensor([], device=pred_se3.device)
        masked_abs_tangle_deg = torch.tensor([], device=pred_se3.device)
        if len(masked_indices) > 0:
            masked_abs_rangle_deg = self.rotation_angle(
                gt_se3[masked_indices, :3, :3], pred_se3[masked_indices, :3, :3]
            )
            masked_abs_tangle_deg = self.translation_angle(
                gt_se3[masked_indices, :3, 3], pred_se3[masked_indices, :3, 3]
            )
        
        # Compute unmasked camera errors
        unmasked_indices = (camera_mask == 0).nonzero(as_tuple=True)[0]
        unmasked_abs_rangle_deg = torch.tensor([], device=pred_se3.device)
        unmasked_abs_tangle_deg = torch.tensor([], device=pred_se3.device)
        if len(unmasked_indices) > 0:
            unmasked_abs_rangle_deg = self.rotation_angle(
                gt_se3[unmasked_indices, :3, :3], pred_se3[unmasked_indices, :3, :3]
            )
            unmasked_abs_tangle_deg = self.translation_angle(
                gt_se3[unmasked_indices, :3, 3], pred_se3[unmasked_indices, :3, 3]
            )
        
        return ((overall_abs_rangle_deg, overall_abs_tangle_deg),
                (masked_abs_rangle_deg, masked_abs_tangle_deg),
                (unmasked_abs_rangle_deg, unmasked_abs_tangle_deg))

    def _evaluate_and_accumulate_auc_metrics(self, predictions, batch, phase):
        """
        Evaluate AUC metrics for the current batch and accumulate for epoch-level computation.
        
        Args:
            predictions: Model predictions containing pose encodings
            batch: Batch data containing ground truth and camera masks
            phase: Current phase ('train' or 'val')
            
        Returns:
            Empty dict (metrics are accumulated internally)
        """
        # Extract predicted pose encodings and convert to extrinsics
        if 'pose_enc' in predictions and predictions['pose_enc'] is not None:
            pred_pose_enc = predictions['pose_enc']
        elif 'pose_enc_list' in predictions and predictions['pose_enc_list']:
            pred_pose_enc = predictions['pose_enc_list'][-1]
        else:
            pred_pose_enc = None
        if pred_pose_enc is None:
            return {}
        
        images = batch['images']
        image_hw = images.shape[-2:]
        
        # Convert predictions to extrinsics and intrinsics
        pred_extrinsic, pred_intrinsic = pose_encoding_to_extri_intri(pred_pose_enc, image_hw)
        gt_extrinsic = batch['extrinsics']
        camera_mask = batch['camera_mask']  # Shape: [B, S] where 1 = masked, 0 = unmasked
        
        batch_size, num_frames = pred_extrinsic.shape[:2]
        
        # Skip if no frames
        if num_frames < 1:
            return {}
        
        # Collect all rotation and translation errors from this batch
        all_rotation_errors = []
        all_translation_errors = []
        all_masked_rotation_errors = []
        all_masked_translation_errors = []
        all_unmasked_rotation_errors = []
        all_unmasked_translation_errors = []
        
        # Process each sequence in the batch
        for b in range(batch_size):
            pred_extri = pred_extrinsic[b]  # (N, 3, 4)
            gt_extri = gt_extrinsic[b]      # (N, 3, 4)
            cam_mask = camera_mask[b]       # (N,) where 1 = masked, 0 = unmasked
            
            # Convert to SE3 format (4x4)
            add_row = torch.tensor([0, 0, 0, 1], device=pred_extri.device).expand(num_frames, 1, 4)
            pred_se3 = torch.cat((pred_extri, add_row), dim=1)
            gt_se3 = gt_extri
            
            # Use absolute pose errors.
            (overall_errors, masked_errors, unmasked_errors) = self.se3_to_absolute_pose_error(
                pred_se3, gt_se3, camera_mask=cam_mask
            )

            # Unpack overall errors
            error_rangle_deg, error_tangle_deg = overall_errors
            
            # Skip if no valid errors (shouldn't happen with our new logic, but defensive)
            if len(error_rangle_deg) == 0:
                continue
            
            # Accumulate overall errors
            all_rotation_errors.append(error_rangle_deg)
            all_translation_errors.append(error_tangle_deg)
            
            # Unpack and accumulate masked errors
            masked_error_rangle_deg, masked_error_tangle_deg = masked_errors
            if len(masked_error_rangle_deg) > 0:
                all_masked_rotation_errors.append(masked_error_rangle_deg)
                all_masked_translation_errors.append(masked_error_tangle_deg)
            
            # Unpack and accumulate unmasked errors
            unmasked_error_rangle_deg, unmasked_error_tangle_deg = unmasked_errors
            if len(unmasked_error_rangle_deg) > 0:
                all_unmasked_rotation_errors.append(unmasked_error_rangle_deg)
                all_unmasked_translation_errors.append(unmasked_error_tangle_deg)
        
        # Accumulate errors if we have valid data
        if all_rotation_errors:
            rotation_errors = torch.cat(all_rotation_errors)
            translation_errors = torch.cat(all_translation_errors)
            
            # Concatenate masked and unmasked errors if they exist
            masked_rotation_errors = torch.cat(all_masked_rotation_errors) if all_masked_rotation_errors else torch.tensor([], device=self.device)
            masked_translation_errors = torch.cat(all_masked_translation_errors) if all_masked_translation_errors else torch.tensor([], device=self.device)
            unmasked_rotation_errors = torch.cat(all_unmasked_rotation_errors) if all_unmasked_rotation_errors else torch.tensor([], device=self.device)
            unmasked_translation_errors = torch.cat(all_unmasked_translation_errors) if all_unmasked_translation_errors else torch.tensor([], device=self.device)
            
            # Determine error type based on number of frames
            error_type = 'absolute' if num_frames == 1 else 'relative'
            
            self._accumulate_auc_metrics(phase, rotation_errors, translation_errors,
                                       masked_rotation_errors, masked_translation_errors,
                                       unmasked_rotation_errors, unmasked_translation_errors,
                                       error_type=error_type)
            
        return {}  # Return empty dict as metrics are accumulated internally
    
def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]

def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
        isinstance(data, Sequence)
        and not isinstance(data, str)
        and len(data) > 0
        and isinstance(data[0], (str, int, float, bool))
    )

def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        if chunk_id == num_chunks - 1:
            # For the last chunk, include all remaining elements
            end = len(data)
        else:
            end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data

