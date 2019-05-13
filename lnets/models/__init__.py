from lnets.models.architectures import *
from lnets.models.model_types import *
from lnets.models.layers import *

MODEL_REGISTRY = {}


def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_model(config):
    model_name = config['model']['name']
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name](config)
    else:
        raise ValueError("Unknown model {:s}".format(model_name))


# Wasserstein Distance Estimation.
@register_model('dual_fc')
def load_fc_dual(config):
    model = FCNet(config.model.layers, config.distrib1.dim, config.model.linear.type, config.model.activation,
                  bias=config.model.linear.bias, config=config)
    return DualOptimModel(model)


@register_model("dual_fully_conv")
def load_conv_dual(config):
    model = FullyConv2D(config.distrib1.dim, config.model.channels, config.model.kernels, config.model.strides,
                        linear_type=config.model.linear.type, activation=config.model.activation, config=config)
    return DualOptimModel(model)


# Classification.
@register_model('classify_fc')
def load_classify_fc(config):
    model = FCNet(config.model.layers, config.data.input_dim, config.model.linear.type, config.model.activation,
                  bias=config.model.linear.bias, config=config)
    return ClassificationModel(model)


@register_model('classify_fc_dropout')
def load_classify_fc_dropout(config):
    model = FCNet(config.model.layers, config.data.input_dim, config.model.linear.type, config.model.activation,
                  bias=config.model.linear.bias, config=config, dropout=True)
    return ClassificationModel(model)


@register_model('classify_fc_spec_jac')
def load_classify_fc_spec_jac(config):
    model = FCNet(config.model.layers, config.data.input_dim, config.model.linear.type, config.model.activation,
                  bias=config.model.linear.bias, config=config)
    return JacSpecClassificationModel(model, config['model']['sn_reg'], config['cuda'])


@register_model('classify_fc_margin')
def load_classify_fc_margin(config):
    model = FCNet(config.model.layers, config.data.input_dim, config.model.linear.type, config.model.activation,
                  bias=config.model.linear.bias, config=config)
    return MarginClassificationModel(model, config)


@register_model('classify_fc_hinge')
def load_classify_fc_hinge(config):
    model = FCNet(config.model.layers, config.data.input_dim, config.model.linear.type, config.model.activation,
                  bias=config.model.linear.bias, config=config)
    return HingeLossClassificationModel(model, config)


@register_model("lenet_classify")
def load_lenet_classify(config):
    model = LeNet(config.data.in_channels, config.model.output_dim, config.model.linear.type, config.model.activation, config.model.dropout_on,
                  config=config)
    return ClassificationModel(model)


@register_model('resnet32')
def CifarResNet32(config):
    block_config = {
        "num_blocks": [5, 5, 5],
        "num_channels": [16, 32, 64],
        "width": 1,
        "pool_size": 8
    }
    return ClassificationModel(ResNet(BasicBlock, block_config, config['data']['class_count']))


@register_model('wide-resnet32')
def CifarWideResNet32(config):
    block_config = {
        "num_blocks": [5, 5, 5],
        "num_channels": [16, 32, 64],
        "width": 10,
        "pool_size": 8
    }
    return ClassificationModel(ResNet(BasicBlock, block_config, config['data']['class_count']))
