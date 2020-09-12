import torch
import torchvision
import torchvision.transforms as transforms


def data_pull_MNIST(Batch_Size=128, Num_Workers=4):
    train_dataset = torchvision.datasets.MNIST('/data/', train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                               ]))

    test_dataset = torchvision.datasets.MNIST('/data/', train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                              ]))

    SEED = 11

    # CUDA?
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)

    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=Batch_Size, num_workers=Num_Workers,
                           pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)

    trainloader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)

    testloader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)


    return trainloader, testloader