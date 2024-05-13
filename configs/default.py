import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.model = "gcnn"
    config.dataset = "cifar10"
    config.lr = 1e-3
    config.num_train_steps = 1000
    config.log_every_steps = 20
    config.save_every_steps = 100
    config.augment_data = True
    config.data_split = (0.8, 0.2)
    config.batch_size = 16

    config.add_noise = False
    config.noise_mean = 0.
    config.noise_var = 1.

    return config