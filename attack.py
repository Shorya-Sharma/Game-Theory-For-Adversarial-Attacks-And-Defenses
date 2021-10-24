from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import cifarresnet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, device, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        trans_data = torch.rand(data.shape, requires_grad=True)
        trans_data = trans_data.to(device)
        for c in range(3):
            trans_data[0, c, :, :] = (data[0, c, :, :] - mean[c]) / std[c]

        # Forward pass the data through the model
        output = model(trans_data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


if __name__ == '__main__':
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    # epsilons = [0]
    # pretrained_model = "models/lenet_mnist_model.pth"
    # pretrained_model = "models/cifar10-resnet20.pth"
    use_cuda = True

    # MNIST Test dataset and dataloader declaration
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #     ])),
    #     batch_size=1, shuffle=True)

    # CIFAR10 Test dataset and dataloader declaration
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.4914, 0.4822, 0.4465],  # mean=[0.5071, 0.4865, 0.4409] for cifar100
            #     std=[0.2023, 0.1994, 0.2010],  # std=[0.2009, 0.1984, 0.2023] for cifar100
            # ),
        ])),
        batch_size=1, shuffle=True)

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # # Initialize the network
    # model = Net().to(device)
    #
    # # Load the pretrained model
    # model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

    #
    model = cifarresnet.cifar_resnet20('cifar10').to(device)

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    accuracies = []
    examples = []

    # Run test for each epsilon
    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    # Plot several examples of adversarial samples at each epsilon
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            # plt.imshow(ex, cmap="gray")
            ex = ex.transpose((1, 2, 0))
            plt.imshow(ex)
    plt.tight_layout()
    plt.show()
