import os
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super().__init__(logdir)

    def add_scalars(self, loss_dict, iteration):
        for key, value in loss_dict.items():
            self.add_scalar(key, value, iteration)

    def add_images(self, tag, images, iteration):
        image = make_grid(images) + 0.5
        self.add_image(tag, image, iteration)


def prepare_logger(output_dir, summary_dir, checkpoint_dir, log_dir):
    for directory in [summary_dir, checkpoint_dir, log_dir]:
        if not os.path.isdir(os.path.join(output_dir, directory)):
            os.makedirs(os.path.join(output_dir, directory))
            os.chmod(os.path.join(output_dir, directory), 0o775)

    logger = Logger(os.path.join(output_dir, summary_dir))
    return logger
