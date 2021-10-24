from torchvision import datasets, transforms
from torch.utils import data


def get_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, 
                                download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=4, 
                                shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, 
                            download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=1, 
                                shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print("train set length = ", len(trainset))
    print("test set length = ", len(testset))
    return trainset, trainloader, testset, testloader, classes
