import torch
import torchvision
import torchvision.transforms as transforms


def data_pull_CIFAR10(Batch_Size=128, Num_Workers=4):
	# Train Phase transformations
	train_transforms = transforms.Compose([
	                                      #  transforms.Resize((28, 28)),
	                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
	                                       transforms.RandomRotation((-10.0, 10.0)),
                                         transforms.RandomHorizontalFlip(),
	                                       transforms.ToTensor(),
	                                       transforms.Normalize((0.4915, 0.4823, 0.4468,), (0.2470, 0.2435, 0.2616,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
	                                       # Note the difference between (0.1307) and (0.1307,)
	                                       ])

	# Test Phase transformations
	test_transforms = transforms.Compose([
	                                      #  transforms.Resize((28, 28)),
	                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
	                                       transforms.ToTensor(),
	                                       transforms.Normalize((0.4915, 0.4823, 0.4468,), (0.2470, 0.2435, 0.2616,))
	                                       ])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
	                                        download=True, transform=train_transforms)


	testset = torchvision.datasets.CIFAR10(root='./data', train=False,
	                                       download=True, transform=test_transforms)

	SEED = 11

	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
	    torch.cuda.manual_seed(SEED)

	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=Batch_Size, num_workers=Num_Workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128)

	trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

	testloader = torch.utils.data.DataLoader(testset, **dataloader_args)

	classes = ('plane', 'car', 'bird', 'cat',
	           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	return trainloader,  testloader, classes