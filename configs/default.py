import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.model = "gcnn"
    config.dataset = "cifar10"
    config.num_classes = 10
    config.lr = 1e-3
    config.num_train_steps = 1000
    config.log_every_steps = 20
    config.save_every_steps = 100

    config.batch_size = 16

    return config