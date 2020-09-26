import torch
import torchvision
import torchvision.transforms as transforms

def data_pull_CIFAR10(train_transforms, test_transforms):

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=train_transforms)


	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=True, transform=test_transforms)

	SEED = 1

	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
	    torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)

	trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

	testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return trainloader,  testloader, classes