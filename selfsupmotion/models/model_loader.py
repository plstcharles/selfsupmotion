import logging
import torch
from torch import optim

from selfsupmotion.models.simsiam import SimSiam

logger = logging.getLogger(__name__)


def load_model(hyper_params, checkpoint):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    architecture = hyper_params['architecture']
    # __TODO__ fix architecture list
    if architecture == 'simsiam':
        model_class = SimSiam
    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))
    if checkpoint is None:
        model = model_class(hyper_params)
    else:
        model = model_class.load_from_checkpoint(checkpoint)
    logger.info('model info:\n' + str(model) + '\n')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('using device {}'.format(device))
    if torch.cuda.is_available():
        logger.info(torch.cuda.get_device_name(0))

    return model
