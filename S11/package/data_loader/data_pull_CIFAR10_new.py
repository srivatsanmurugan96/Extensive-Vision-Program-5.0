import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
import pprint
from data_augmentation.albumenatiton import albumentations_transforms
from utils.summary import has_cuda

class data_pull_CIFAR10(object):
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
			   'frog', 'horse', 'ship', 'truck')

	def __init__(self, args):
		super(data_pull_CIFAR10, self).__init__()
		self.batch_size_cuda = args.batch_size_cuda
		self.batch_size_cpu = args.batch_size_cpu
		self.num_workers = args.num_workers
		self.load()

	def _transforms(self):
		# Data Transformations
		train_transform = albumentations_transforms(p=1.0, is_train=True)
		test_transform = albumentations_transforms(p=1.0, is_train=False)
		return train_transform, test_transform

	def _dataset(self):
		# Get data transforms
		train_transform, test_transform = self._transforms()

		# Dataset and Creating Train/Test Split
		train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
												 download=True, transform=train_transform)
		test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
												download=True, transform=test_transform)
		return train_set, test_set

	def load(self):
		# Get Train and Test Data
		train_set, test_set = self._dataset()

		# Dataloader Arguments & Test/Train Dataloaders
		dataloader_args = dict(
			shuffle=True,
			batch_size=self.batch_size_cpu)
		if has_cuda():
			dataloader_args.update(
				batch_size=self.batch_size_cuda,
				num_workers=self.num_workers,
				pin_memory=True)

		self.train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
		self.test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)

