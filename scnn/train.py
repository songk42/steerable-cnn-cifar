from absl import app
from absl import flags
from absl import logging
import ml_collections
from ml_collections import config_flags
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from scnn.models import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_bool("use_wandb", True, "Whether to log to Weights & Biases.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def train(config, workdir):
    device = 0
    classifier = utils.create_model(config).to(device)
    classifier.train()
    optimizer = optim.Adam(classifier.parameters(), lr=config.lr)

    train_loader, test_loader = utils.load_data(config)

    logging.info("Training model.")

    times = []
    for step in range(config.num_train_steps):
        batch_losses = []

        start_time = time.time()
        for data in train_loader:
            x = data[0].to(device)
            y = data[1].to(device)
            output = classifier(x)
            loss = nn.functional.cross_entropy(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
        times.append(time.time() - start_time)

        if step % config.log_every_steps == 0:
            logging.info(f"Step {step}: avg loss {torch.mean(torch.tensor(batch_losses))} ({np.mean(np.asarray(times)):.2f}s per step)")
        if step % config.save_every_steps == 0:
            torch.save(classifier.state_dict(), os.path.join(workdir, f"model_{step}.pth"))

    logging.info("Finished training.")

    logging.info("Evaluating model.")
    classifier.eval()
    correct = 0
    total = 0
    confusion = torch.zeros(config.num_classes, config.num_classes)
    with torch.no_grad():
        for data in test_loader:
            x = data[0].to(device)
            y = data[1].to(device)
            output = classifier(x)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            for i in range(y.size(0)):
                confusion[y[i], predicted[i]] += 1
    accuracy = correct / total
    logging.info(f"Accuracy: {correct} / {total} ({accuracy})")
    logging.info(f"Confusion matrix (true, predicted): {confusion}")

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("Local devices: %r", torch.cuda.device_count())
    logging.info("CUDA_VISIBLE_DEVICES: %r", os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Check and freeze config.
    config = FLAGS.config
    config = ml_collections.FrozenConfigDict(config)

    # Initialize wandb.
    if FLAGS.use_wandb:
        os.makedirs(os.path.join(FLAGS.workdir, "wandb"), exist_ok=True)

        wandb.login()
        wandb.init(
            project="scnn",
            config=config.to_dict(),
            dir=FLAGS.workdir,
        )

    # Start training!
    train(config, FLAGS.workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
