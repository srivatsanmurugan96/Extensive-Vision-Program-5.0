from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# train_losses = []
# test_losses = []
# train_acc = []
# test_acc = []


def train(model, device, train_loader, criterion, optimizer, epoch,
          l1_decay, l2_decay, train_losses, train_accs, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    avg_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
    # get samples
        data, target = data.to(device), target.to(device)

    # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)
        if l1_decay > 0:
          l1_loss = 0
          for param in model.parameters():
            l1_loss += torch.norm(param,1)
          loss += l1_decay * l1_loss
        if l2_decay > 0:
          l2_loss = 0
          for param in model.parameters():
            l2_loss += torch.norm(param,2)
          loss += l2_decay * l2_loss

        # Backpropagation
        loss.backward()
        optimizer.step()
        if scheduler:
          scheduler.step()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        avg_loss += loss.item()

        pbar_str = f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
        if l1_decay > 0:
          pbar_str = f'L1_loss={l1_loss.item()} %s' % (pbar_str)
        if l2_decay > 0:
          pbar_str = f'L2_loss={l2_loss.item()} %s' % (pbar_str)

        pbar.set_description(desc= pbar_str)

    avg_loss /= len(train_loader)
    avg_acc = 100*correct/processed
    train_accs.append(avg_acc)
    train_losses.append(avg_loss)


def test(model, device, test_loader, criterion, classes, test_losses, test_accs,
         misclassified_imgs, correct_imgs, is_last_epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            is_correct = pred.eq(target.view_as(pred))
            if is_last_epoch:
                misclassified_inds = (is_correct == 0).nonzero()[:, 0]
                for mis_ind in misclassified_inds:
                    if len(misclassified_imgs) == 25:
                        break
                    misclassified_imgs.append({
                        "target": target[mis_ind].cpu().numpy(),
                        "pred": pred[mis_ind][0].cpu().numpy(),
                        "img": data[mis_ind]
                    })

                correct_inds = (is_correct == 1).nonzero()[:, 0]
                for ind in correct_inds:
                    if len(correct_imgs) == 25:
                        break
                    correct_imgs.append({
                        "target": target[ind].cpu().numpy(),
                        "pred": pred[ind][0].cpu().numpy(),
                        "img": data[ind]
                    })
            correct += is_correct.sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    test_acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(test_acc)

    if test_acc >= 90.0:
        classwise_acc(model, device, test_loader, classes)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))


def classwise_acc(model, device, test_loader, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # print class-wise test accuracies
    print()
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    print()


# def run(l1_decay=0.0, l2_decay=0.0):
def fit(model, device, train_loader, test_loader,classes,criterion,optimizer,scheduler , epochs, l1_decay=0.0, l2_decay=0.0):
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs,
    #                       steps_per_epoch=len(train_loader))
    test_losses = []
    test_accs = []
    train_losses = []
    train_accs = []
    misclassified_imgs = []
    correct_imgs = []
    for epoch in range(epochs):
        print("EPOCH:", epoch + 1)
        train(model, device, train_loader, criterion, optimizer, epoch,
              l1_decay, l2_decay, train_losses, train_accs)
        test(model, device, test_loader, criterion, classes, test_losses, test_accs,
             misclassified_imgs, correct_imgs, epoch == epochs - 1)
        scheduler.step(test_losses[-1])
    return train_losses, train_accs, test_losses, test_accs, misclassified_imgs, correct_imgs

#######################################################################


# def train(model, device, train_loader, criterion, optimizer, epoch,
#           l1_decay, l2_decay, scheduler=None):
#   model.train()
#   pbar = tqdm(train_loader)
#   correct = 0
#   processed = 0
#   for batch_idx, (data, target) in enumerate(pbar):
#     # get samples
#     data, target = data.to(device), target.to(device)
#
#     # Init
#     optimizer.zero_grad()
#     # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
#     # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.
#
#     # Predict
#     y_pred = model(data)
#
#     # Calculate loss
#     loss = criterion(y_pred, target)
#     if l1_decay > 0:
#       l1_loss = 0
#       for param in model.parameters():
#         l1_loss += torch.norm(param,1)
#       loss += l1_decay * l1_loss
#     if l2_decay > 0:
#       l2_loss = 0
#       for param in model.parameters():
#         l2_loss += torch.norm(param,2)
#       loss += l2_decay * l2_loss
#
#     # Backpropagation
#     loss.backward()
#     optimizer.step()
#     if scheduler:
#       scheduler.step()
#
#     # Update pbar-tqdm
#     pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#     correct += pred.eq(target.view_as(pred)).sum().item()
#     processed += len(data)
#
#     pbar_str = f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}'
#     if l1_decay > 0:
#       pbar_str = f'L1_loss={l1_loss.item()} %s' % (pbar_str)
#     if l2_decay > 0:
#       pbar_str = f'L2_loss={l2_loss.item()} %s' % (pbar_str)
#
#     pbar.set_description(desc= pbar_str)
#
#
# def test(model, device, test_loader, criterion, classes, test_losses, test_accs,
#          misclassified_imgs, correct_imgs, is_last_epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             is_correct = pred.eq(target.view_as(pred))
#             if is_last_epoch:
#                 misclassified_inds = (is_correct == 0).nonzero()[:, 0]
#                 for mis_ind in misclassified_inds:
#                     if len(misclassified_imgs) == 25:
#                         break
#                     misclassified_imgs.append({
#                         "target": target[mis_ind].cpu().numpy(),
#                         "pred": pred[mis_ind][0].cpu().numpy(),
#                         "img": data[mis_ind]
#                     })
#
#                 correct_inds = (is_correct == 1).nonzero()[:, 0]
#                 for ind in correct_inds:
#                     if len(correct_imgs) == 25:
#                         break
#                     correct_imgs.append({
#                         "target": target[ind].cpu().numpy(),
#                         "pred": pred[ind][0].cpu().numpy(),
#                         "img": data[ind]
#                     })
#             correct += is_correct.sum().item()
#
#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)
#
#     test_acc = 100. * correct / len(test_loader.dataset)
#     test_accs.append(test_acc)
#
#     if test_acc > 85.0:
#         classwise_acc(model, device, test_loader, classes)
#
#     print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset), test_acc))
#
#
# def classwise_acc(model, device, test_loader, classes):
#     class_correct = list(0. for i in range(10))
#     class_total = list(0. for i in range(10))
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(4):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1
#
#     # print class-wise test accuracies
#     print()
#     for i in range(10):
#         print('Accuracy of %5s : %2d %%' % (
#             classes[i], 100 * class_correct[i] / class_total[i]))
#     print()



# def fit(model, device, train_loader, test_loader,classes,criterion,optimizer,scheduler , epochs, l1_decay=0.0, l2_decay=0.0):
#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#     # scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs,
#     #                       steps_per_epoch=len(train_loader))
#     test_losses = []
#     test_accs = []
#     misclassified_imgs = []
#     correct_imgs = []
#     for epoch in range(epochs):
#       print("EPOCH:", epoch+1)
#       train(model, device, train_loader, criterion, optimizer, epoch, l1_decay, l2_decay, scheduler)
#       test(model, device, test_loader, criterion, classes, test_losses, test_accs,
#            misclassified_imgs, correct_imgs, epoch==epochs-1)
#     return test_losses, test_accs, misclassified_imgs, correct_imgs

