from typing import Literal, Union
import torch.nn as nn

LayerType = Union[int, Literal["M"]]


class CifarVGG(nn.Module):
    """Base class for convolutional neural network for CIFAR-style datasets based on the VGG
    architectures.
    """

    def __init__(
        self,
        plan: list[LayerType],
        input_channels: int = 3,
        num_classes: int = 10,
        readout_width: int = 128,
        readout_depth: int = 1,
    ):
        super(CifarVGG, self).__init__()
        self.plan = plan

        # Build convolutional layers
        layers = []
        in_channels = input_channels

        for layer in plan:
            if layer == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = layer
                layers.extend(
                    [
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),  # Added BatchNorm as per Part 2 requirements
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Build classifier (MLP readout)
        classifier_layers = []
        in_features = 512 * 2 * 2  # After conv layers, output is (512, 2, 2)

        # Hidden layers
        for _ in range(readout_depth):
            classifier_layers.extend(
                [
                    nn.Linear(in_features, readout_width),
                    nn.ReLU(inplace=True),
                    nn.Dropout(),  # Helps prevent overfitting
                ]
            )
            in_features = readout_width

        # Final output layer
        classifier_layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class VGG11(CifarVGG):
    PLAN = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]

    def __init__(self, **kwargs):
        super().__init__(self.PLAN, **kwargs)


class VGG13(CifarVGG):
    PLAN = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512]

    def __init__(self, **kwargs):
        super().__init__(self.PLAN, **kwargs)


class VGG16(CifarVGG):
    PLAN = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]

    def __init__(self, **kwargs):
        super().__init__(self.PLAN, **kwargs)


class VGG19(CifarVGG):
    PLAN = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ]  # noqa: E501

    def __init__(self, **kwargs):
        super().__init__(self.PLAN, **kwargs)


def get_model(name: str) -> CifarVGG:
    name = name.upper()
    if name == "VGG11":
        return VGG11()
    elif name == "VGG13":
        return VGG13()
    elif name == "VGG16":
        return VGG16()
    elif name == "VGG19":
        return VGG19()
    else:
        raise ValueError(f"Unknown model {name}")


__all__ = ["VGG11", "VGG13", "VGG16", "VGG19", "get_model", "LayerType"]
