"""
nnUNetTrainerMCDropout — adds Dropout3d to every encoder/decoder conv block.

How to use:
  python nnunet_run.py nnUNetv2_train 1 3d_fullres 0 -tr nnUNetTrainerMCDropout

nnU-Net discovers custom trainers via PYTHONPATH.  Run from repo root:
  set PYTHONPATH=G:\\THO_space\\TEST\\vasc\\src
  python nnunet_run.py nnUNetv2_train ...

Or use src/train_mc_dropout.py which sets PYTHONPATH automatically.

Design notes:
  PlainConvUNet already accepts dropout_op + dropout_op_kwargs in its __init__.
  The nnU-Net plan sets these to None by default.  We override
  build_network_architecture() to inject Dropout3d before the network is built —
  no monkey-patching, no post-hoc surgery, no forking nnU-Net.

  At inference time, call model.train() before each forward pass to keep dropout
  active.  See src/inference_mc.py for the T=20 MC Dropout inference loop.
"""
import copy

import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainerMCDropout(nnUNetTrainer):
    """
    Identical to nnUNetTrainer except Dropout3d (p=0.1) is added to every
    encoder and decoder conv block via the architecture's native dropout_op hook.

    Dropout is active during training (standard) and kept active at inference
    by calling model.train() — see inference_mc.py.
    """

    dropout_p: float = 0.1

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision=True) -> nn.Module:
        arch_kwargs = copy.deepcopy(configuration_manager.network_arch_init_kwargs)
        arch_kwargs_req_import = list(
            configuration_manager.network_arch_init_kwargs_req_import
        )

        # Inject Dropout3d — the architecture resolves the string via pydoc.locate
        arch_kwargs["dropout_op"] = "torch.nn.modules.dropout.Dropout3d"
        arch_kwargs["dropout_op_kwargs"] = {"p": nnUNetTrainerMCDropout.dropout_p}

        # dropout_op must be in req_import so pydoc resolves the class string
        if "dropout_op" not in arch_kwargs_req_import:
            arch_kwargs_req_import.append("dropout_op")

        return get_network_from_plans(
            configuration_manager.network_arch_class_name,
            arch_kwargs,
            arch_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
