from .cyclegan_trainer import CycleGANTrainer
from .cartoongan_trainer import CartoonGanTrainer
from .whitebox_trainer import WhiteboxTrainer


def build_trainer(config):
    if config.exp_name == 'cyclegan':
        trainer = CycleGANTrainer(config)
    elif config.exp_name == 'cartoongan':
        trainer = CartoonGanTrainer(config)
    elif config.exp_name == 'whitebox':
        trainer = WhiteboxTrainer(config)
    else:
        raise NotImplementedError
    return trainer