import copy
from mmcv.utils import build_from_cfg
from mmcv.runner import build_optimizer_constructor, OPTIMIZER_BUILDERS
from mmcv.runner import DefaultOptimizerConstructor, OPTIMIZERS


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer


@OPTIMIZER_BUILDERS.register_module()
class DGOptimizerConstructor(DefaultOptimizerConstructor):
    def __call__(self, model):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()

        param_dict = dict(model.named_parameters())
        params = []
        for key, value in param_dict.items():
            if 'mask' in key:
                params += [{'params': [value], 'lr': optimizer_cfg['lr'], 'weight_decay': 0.}]
            else:
                params += [{'params':[value]}]

        optimizer_cfg['params'] = params
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
