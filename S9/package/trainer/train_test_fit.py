from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim

train_losses = []
test_losses = []
train_acc = []
test_acc = []


def train(model, device, train_loader, optimizer, epoch, lambda_l1, criterion, train_losses, train_acc):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)

        # L1 Regularization
        l1 = 0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
        loss = loss + (lambda_l1 * l1)

        loss.backward()
        optimizer.step()

        processed += len(data)

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

        pbar.set_description('Train: Batch id: {} \tLoss: {:.6f}\t Accuracy:{:.3f}'.format(
            batch_idx, loss.item(), 100 * correct / processed))

        train_losses.append(loss.item())
        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).sum().item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: loss: {:.6f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(
        100. * correct / len(test_loader.dataset))

def fit(model, device, train_loader,test_loader, epochs,l1=0,l2=0):
  lambda_l1 = l1
  lambda_l2 = l2

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=lambda_l2)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)

  train_losses = []
  test_losses = []
  train_accuracy = []
  test_accuracy = []

  for epoch in range(epochs):
      print("EPOCH:", epoch)
      train(model, device, train_loader, optimizer,epoch, lambda_l1,criterion,train_losses, train_acc)
      scheduler.step()
      test(model, device, test_loader,criterion, test_losses, test_acc)

def Class_Accuracy(device, model, testloader, classes):
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images =images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs, 1)
          c = (predicted == labels).squeeze()
          for i in range(4):
              label = labels[i]
              class_correct[label] += c[i].item()
              class_total[label] += 1


  for i in range(10):
      print('Accuracy of %5s : %2d %%' % (
          classes[i], 100 * class_correct[i] / class_total[i]))