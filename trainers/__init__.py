from .cyclegan_trainer import CycleGANTrainer


def build_trainer(config):
    if config.exp_name == 'cyclegan':
        trainer = CycleGANTrainer(config)
    elif config.exp_name == 'cartoongan':
        raise NotImplementedError
    elif config.exp_name == 'whitebox':
        raise NotImplementedError
    else:
        raise NotImplementedError
    return trainer