from .cyclegan_trainer import CycleGANTrainer
from .cartoongan_trainer import CartoonGanTrainer
from .whitebox_trainer import WhiteboxTrainer
from .stargan_trainer import StarCartoonTrainer
from .classifier_trainer import ClassifierTrainer


def build_trainer(config):
    if config.exp_name == 'cyclegan':
        trainer = CycleGANTrainer(config)
    elif config.exp_name == 'cartoongan':
        trainer = CartoonGanTrainer(config)
    elif config.exp_name == 'whitebox':
        trainer = WhiteboxTrainer(config)
    elif config.exp_name == 'stargan':
        trainer = StarCartoonTrainer(config)
    elif config.exp_name == 'classifier':
        trainer = ClassifierTrainer(config)
    else:
        raise NotImplementedError
    return trainer