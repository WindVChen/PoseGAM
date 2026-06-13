# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2025 WindVChen.
# All rights reserved.
#
# This source code is derived from VGGT and licensed under the VGGT License
# found in the LICENSE_VGGT file in the root directory of this source tree.


from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from posegam.training.trainer import Trainer


with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="default")      # loads config/default.yaml

trainer = Trainer(**cfg)
trainer.run()


