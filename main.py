from __future__ import print_function

import argparse
import os
from pathlib import Path
from matplotlib import image
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import numpy.matlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import random

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''


class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''

    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.dp1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.dp2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(16,32,3,padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.dp3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(288, 64) #576 3*3*channel
        self.fc2 = nn.Linear(64, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        #x = self.dp1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        #

        x = self.conv2(x)
        x = F.relu(x)
        #x = self.dp2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        #

        x = self.conv3(x)
        x = F.relu(x)
        #x = self.dp3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, 2)


        # print(x.size())
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        feature_vector = x
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x

    def extract_feature(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        #x = self.dp1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        #

        x = self.conv2(x)
        x = F.relu(x)
        #x = self.dp2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        #

        x = self.conv3(x)
        x = F.relu(x)
        #x = self.dp3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, 2)


        # print(x.size())
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        feature_vector = x

        return feature_vector


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()  # Set the model to training mode
    train_loss = 0
    correct = 0
    train_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear the gradient
        output = model(data)  # Make predictions
        loss = F.nll_loss(output, target)  # Compute loss
        train_loss += loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_num += len(data)
        loss.backward()  # Gradient computation
        optimizer.step()  # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_loss = train_loss / len(train_loader.sampler)
    train_accuracy = correct / train_num

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, train_num, 100. * train_accuracy))
    return train_loss, train_accuracy


def test(model, device, test_loader):
    model.eval()  # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    preds = []
    gts = []
    with torch.no_grad():  # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

            # for
            preds += list(np.asarray(pred))
            gts += list(np.asarray(target))
            '''
            # save the misclassified data
            mis_indices = (pred.eq(target.view_as(pred)) == False).nonzero()[:,0]
            pre_label = np.asarray(pred[mis_indices])
            gt = target[mis_indices]
            mis_data = data[mis_indices]

            for i, im in enumerate(mis_data):
                im = np.asarray(im)
                im = im.transpose(1,2,0)
                im = (im-np.min(im))/(np.max(im)-np.min(im))
                image.imsave(Path('../data/hw03/mis_classify{}_as{}.png'.format(gt[i],pre_label[i])), im[...,0])
                #Image.fromarray(im[...,0]).save(Path('../data/hw03/mis_classify{}_as{}.png'.format(gt[i],pre_label[i])))
            '''
    test_loss /= test_num
    test_accuracy = correct / test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num, 100. * test_accuracy))

    print(confusion_matrix(gts, preds))
    return test_loss, test_accuracy

def feature_maps_embedding(model, device, test_loader):
    # for i, layer in enumerate(model.features):
    #     input_tensor = layer(input_tensor)
    #
    # feature_vector = input_tensor
    # return feature_vector
    model.eval()  # Set the model to inference mode

    tsne = TSNE(n_components=2, random_state=0)
    with torch.no_grad():  # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            feature_maps = model.extract_feature(data)
            y = target

    X_2d = tsne.fit_transform(feature_maps)
    target_ids = range(10)
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, target_ids):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.title('Mnist high-dimensional embedding in 2D using tSNE')
    plt.savefig(Path('../data/hw03/embedding.png'),bbox = 'tight')

    for i in range(8):
        d = np.sqrt(np.sum((np.asarray(feature_maps) - np.matlib.repmat(feature_maps[i],feature_maps.shape[0],1))**2,1))
        idx = np.argsort(d)
        for j in range(9):
            im = data[idx[j]]
            im = np.asarray(im)
            im = im.transpose(1,2,0)
            im = (im-np.min(im))/(np.max(im)-np.min(im))
            image.imsave(Path('../data/hw03/embedding{}_{}th_closeset.png'.format(i,j)), im[...,0])
def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='evaluate your model on the official test set')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='visulaze the kernel')
    parser.add_argument('--load-model', type=str, default='mnist_model.pt',
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                                      transform=transforms.Compose([
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=True, **kwargs) #args.test_batch_size
        feature_maps_embedding(model, device, test_loader)
        #test(model, device, test_loader)

        return
    # Visualize the kernels
    # Evaluate on the official test set
    if args.visualize:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        kernels = model.conv1.weight.data
        for i,kernel in enumerate(kernels):
            kernel = np.asarray(kernel)[0,...]
            image.imsave(Path('../data/hw03/kernel{}.png'.format(i)), kernel)

        return


    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=False,
                                   transform=transforms.Compose([  # Data preprocessing
                                       transforms.RandomRotation(10),
                                       transforms.ToTensor(),  # Add data augmentation here
                                       #transforms.RandomErasing(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.

    # subset_indices_train = []
    # subset_indices_valid = []
    #
    # np.random.seed(148)
    # validation_split = .15
    # for c in range(10):
    #     indices = []
    #     for i, data in enumerate(train_dataset):
    #         if data[1] == c:
    #             indices.append(i)
    #
    #     split = int(np.floor(validation_split * len(indices)))
    #     np.random.shuffle(indices)
    #     train_indices, val_indices = indices[split:], indices[:split]
    #     subset_indices_train += train_indices
    #     subset_indices_valid += val_indices
    #
    # np.save(Path('../data/hw03_splited_indices/splited_indices.npy'),np.asarray([subset_indices_train,subset_indices_valid]))
    # print('saved!')
    subset_indices_train, subset_indices_valid = np.load(Path('../data/hw03/splited_indices.npy'),allow_pickle=True)

    #answer_q7(args, subset_indices_train_original, subset_indices_valid, train_dataset, device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, val_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()  # learning rate scheduler

        # You may optionally save your model at each epoch here

    # Plot training and val loss as a function of the epoch. Use this to monitor for overfitting.
    plt.plot(range(1, args.epochs + 1), train_losses, label='traning')
    plt.plot(range(1, args.epochs + 1), test_losses, label='testing')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.savefig(Path('../data/loss.png'), bbox_inches='tight')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model.pt")

def answer_q7(args, subset_indices_train_original, subset_indices_valid, train_dataset, device):
    sizes = [1/2,1/4,1/8,1/16]
    traning_error = []
    test_error = []
    for size in sizes:
        print(size)
        subset_indices_train  = random.sample(subset_indices_train_original, int(size*len(subset_indices_train_original)))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_train)
        )
        val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.test_batch_size,
            sampler=SubsetRandomSampler(subset_indices_valid)
        )

        # Load your model [fcNet, ConvNet, Net]
        model = Net().to(device)

        # Try different optimzers here [Adam, SGD, RMSprop]
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        # Set your learning rate scheduler
        scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

        # Training loop
        for epoch in range(1, args.epochs + 1):
            train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
            test_loss, test_accuracy = test(model, device, val_loader)
            scheduler.step()  # learning rate scheduler

            # You may optionally save your model at each epoch here
        traning_error.append(1-train_accuracy)
        test_error.append(1-test_accuracy)

    xs = [int(size * len(subset_indices_train_original)) for size in sizes]
    plt.loglog(xs, traning_error, label='traning')
    plt.loglog(xs, test_error, label='testing')
    plt.legend(loc='best')
    plt.xlabel('training examples')
    plt.ylabel('Error')
    plt.title(' Training and test error on log-log scale')
    plt.savefig(Path('../data/error_loglog.png'), bbox_inches='tight')



if __name__ == '__main__':
    main()
