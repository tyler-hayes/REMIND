import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18


def get_name_to_module(model):
    name_to_module = {}
    for m in model.named_modules():
        name_to_module[m[0]] = m[1]
    return name_to_module


def get_activation(all_outputs, name):
    def hook(model, input, output):
        all_outputs[name] = output.detach()

    return hook


def add_hooks(model, outputs, output_layer_names):
    """

    :param model:
    :param outputs: Outputs from layers specified in `output_layer_names` will be stored in `output` variable
    :param output_layer_names:
    :return:
    """
    name_to_module = get_name_to_module(model)
    for output_layer_name in output_layer_names:
        name_to_module[output_layer_name].register_forward_hook(get_activation(outputs, output_layer_name))


class ModelWrapper(nn.Module):
    def __init__(self, model, output_layer_names, return_single=False):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_layer_names = output_layer_names
        self.outputs = {}
        self.return_single = return_single
        add_hooks(self.model, self.outputs, self.output_layer_names)

    def forward(self, images):
        self.model(images)
        output_vals = [self.outputs[output_layer_name] for output_layer_name in self.output_layer_names]
        if self.return_single:
            return output_vals[0]
        else:
            return output_vals


def test_resnet18():
    output_layer_names = ['layer1.0.bn1', 'layer4.0', 'fc']
    in_tensor = torch.ones((2, 3, 224, 224))

    core_model = resnet18()
    wrapper = ModelWrapper(core_model, output_layer_names)
    y1, y2, y3 = wrapper(in_tensor)
    assert y1.shape[0] == 2
    assert y1.shape[2] == 56
    assert y2.shape[2] == 7
    assert y3.shape[1] == 1000


if __name__ == "__main__":
    test_resnet18()
