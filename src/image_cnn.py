import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    simple neural network model.
    """
    def __init__(self, n_classes):
        super(Net, self).__init__()
        # 1 input channel, 32 output channel, 5x5 square kernel
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=5)

        self.pool = nn.MaxPool2d(kernel_size=2,
                                 stride=2)

        self.fc1 = nn.Linear(in_features=32*4,
                             out_features=n_classes)

    # define the feed-forward behavior
    def forward(self, x):
        # one conv/relu + max-pooling
        x = self.pool(F.relu(self.conv1(x)))

        # prep for linear layer and flattening the output
        x = x.view(x.size(0), -1)

        # linear output layer
        x = F.relu(self.fc1(x))

        return x

    def backward(self):
        pass


if __name__ == "__main__":
    n_classes = 20
    net = Net(n_classes)
    print(net)
