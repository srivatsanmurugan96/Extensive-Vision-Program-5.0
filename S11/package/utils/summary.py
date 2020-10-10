import torch
from torchsummary import summary

def has_cuda():
	return torch.cuda.is_available()

def get_summary(model, input):
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	model = model.to(device)
	return summary(model, input_size=input)