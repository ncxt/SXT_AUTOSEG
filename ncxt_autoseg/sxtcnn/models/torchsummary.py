import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

#  adapted from https://github.com/sksq96/pytorch-summary/tree/master/torchsummary


def size_mb(model, input_size, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    # print(x.shape)
    model(x.to(device))
    # remove these hooks
    for h in hooks:
        h.remove()

    bits = 32

    total_params = 0
    trainable_params = 0
    total_size = 0
    for layer in summary:
        shapes = summary[layer]["output_shape"]

        if isinstance(shapes[0], int):
            total_size += int(np.prod(shapes[1:]))
        else:
            for subshape in shapes:
                total_size += int(np.prod(subshape[1:]))

        total_params += summary[layer]["nb_params"]
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

    inp_bits = int(np.prod(input_size))
    param_bits = total_params
    intern_bits = (total_size * 2)
    total = inp_bits + param_bits + intern_bits
    return (total * bits / 8 / (1024 ** 2))


def summary(model, input_size, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # print(type(x[0]))
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    # print(x.shape)
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    bits = 32

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    trainable_params = 0
    total_size = 0
    for layer in summary:
        shapes = summary[layer]["output_shape"]

        if isinstance(shapes[0], int):
            total_size += int(np.prod(shapes[1:]))
        else:
            for subshape in shapes:
                total_size += int(np.prod(subshape[1:]))

        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    print("================================================================")
    print("Total size: {0:,}".format(total_size))
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))

    inp_bits = int(np.prod(input_size))
    param_bits = total_params
    intern_bits = 2*total_size 
    total = inp_bits + param_bits + intern_bits
    print(f"Input size: {inp_bits}")
    print(f"params size: {total_params}")
    print(f"Internal size: {intern_bits}")

    print(f"Input size: {bytes_2_human_readable(inp_bits* bits/8)}")
    print(f"Params size: {bytes_2_human_readable(param_bits* bits/8)}")
    print(f"Total size: {bytes_2_human_readable(total* bits/8)}")
    print("----------------------------------------------------------------")
    # return summary

    # print(len(summary[layer]['output_shape']))
    # print(summary[layer]['output_shape'])


def bytes_2_human_readable(number_of_bytes, precision=2):
    if number_of_bytes < 0:
        raise ValueError("!!! number_of_bytes can't be smaller than 0 !!!")

    step_to_greater_unit = 1024.0
    number_of_bytes = float(number_of_bytes)
    unit_list = ["bytes", "KB", "MB", "GB", "TB"]

    for unit in unit_list:
        if number_of_bytes / step_to_greater_unit < 1:
            return str(round(number_of_bytes, precision)) + " " + unit
        number_of_bytes /= step_to_greater_unit

    number_of_bytes *= step_to_greater_unit
    return str(round(number_of_bytes, precision)) + " " + unit


if __name__ == "__main__":

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=5)
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3)

        def forward(self, x):
            h = self.conv0(x)
            h = self.conv1(h)
            return h

    model = Model()
    summary(model, (1, 32, 32), device="cpu")

