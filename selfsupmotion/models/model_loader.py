import logging
import torch

import selfsupmotion.models.kpts_regressor
import selfsupmotion.models.simsiam

logger = logging.getLogger(__name__)


def load_model(hyper_params):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    architecture = hyper_params['architecture']
    # __TODO__ fix architecture list
    if architecture == 'simsiam':
        model_class = selfsupmotion.models.simsiam.SimSiam
    elif architecture == 'kpts_regressor':
        model_class = selfsupmotion.models.kpts_regressor.KeypointsRegressor
    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))

    model = model_class(hyper_params)
    logger.info('model info:\n' + str(model) + '\n')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('using device {}'.format(device))
    if torch.cuda.is_available():
        logger.info(torch.cuda.get_device_name(0))

    return model
