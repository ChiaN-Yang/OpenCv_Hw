from PyQt5 import QtWidgets, uic
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
from torchinfo import summary
import torch.nn as nn
import cv2
from torch.autograd import Variable


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Load the UI Page
        uic.loadUi('./doc/opencvdl_hw1_5.ui', self)
        # self.pushButton_0.clicked.connect(self.startTrainModel)
        self.pushButton_1.clicked.connect(self.showTrainImages)
        self.pushButton_2.clicked.connect(self.showHyperParameter)
        self.pushButton_3.clicked.connect(self.showModelShortcut)
        self.pushButton_4.clicked.connect(self.showAccuracy)
        self.pushButton_5.clicked.connect(self.test)

        # Load and normalize CIFAR10
        size = 224
        train_transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.ToTensor(),
             # this parameters are mean and std of Imagenet
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        test_transform = transforms.Compose(
            [transforms.Resize(size),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.batch_size = 32
        self.lr = 0.001

        download = not os.path.isdir(r'./Dataset_OpenCvDl_Hw1/Q5_Image')
        self.trainset = torchvision.datasets.CIFAR10(root='./Dataset_OpenCvDl_Hw1/Q5_Image', train=True,
                                                     download=download, transform=train_transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                       shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./Dataset_OpenCvDl_Hw1/Q5_Image', train=False,
                                                    download=download, transform=test_transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def showTrainImages(self):
        plt.figure()

        for i, (img, label) in enumerate(self.trainloader):
            if i >= 9:
                break

            # get some random training images
            img = img / 2 + 0.5     # unnormalize
            img = img[0]
            npimg = img.numpy()

            plt.subplot(3, 3, i+1)
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.axis('off')
            plt.title(self.classes[label[0]])
        plt.show()

    def showHyperParameter(self):
        print('\nhyperparameters:')
        print(f'batch Size: {self.batch_size}')
        print(f'learning rate: {self.lr}')
        print(f'optimizer: SGD\n')

    def showModelShortcut(self):
        model = models.vgg16(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        for x_batch, _ in self.trainloader:
            break

        input_shape = x_batch.shape

        summary(model, input_shape)
        print(model)
        print("==========================================================================================")

    def showAccuracy(self):
        self.img = cv2.imread("./doc/training_loss_and_accuracy.png")
        cv2.namedWindow('training loss and accuracy', cv2.WINDOW_NORMAL)
        cv2.imshow('training loss and accuracy', self.img)

    def test(self):
        x = int(self.lineEdit.text())
        self.inferenceImage(x)

    def inferenceImage(self, x):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        softmax = nn.Softmax(dim=0)

        # Load the most accurate model
        model = models.vgg16()
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        # fine tune the last classifier layer from 1000 to 2 dim(num. of classifier).
        model.classifier[6] = nn.Linear(4096, 10)
        model.load_state_dict(torch.load(
            './doc/VGG16_model.pth', map_location=device))
        with torch.no_grad():
            model.eval()

            x_batch, y_batch = self.testset[x]
            img = x_batch
            x_batch = x_batch.to(device)
            x_batch = Variable(torch.unsqueeze(
                x_batch, dim=0).float(), requires_grad=False)
            p_batch = model(x_batch)
            probs = torch.relu(p_batch[0])
            probs = softmax(probs)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        img = img / 2 + 0.5     # unnormalize
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.bar(self.classes, probs)
        plt.xlabel("Classes")
        plt.ylabel("Probability")

        plt.show()

    def startTrainModel(self):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        model = models.vgg16(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        # fine tune the last classifier layer from 1000 to 2 dim(num. of classifier).
        model.classifier[6] = nn.Linear(4096, 10)

        model = model.to(device)                      # load model to GPU
        loss_func = nn.CrossEntropyLoss()                      # set loss function
        optimizer = optim.SGD(model.parameters(), lr=self.lr,
                              momentum=0.9, weight_decay=5e-4)
        epochs = 20

        # create empty lists for saving metrics during training
        train_loss_list = []
        train_accuracy_list = []
        valid_loss_list = []
        valid_accuracy_list = []
        valid_loss_min = np.Inf  # track change in validation loss

        for epoch in range(epochs):

            print(f"Epoch {epoch+1}/{epochs}")

            # initialize metrics value
            train_correct_count = 0
            train_accuracy = 0
            train_loss = 0
            valid_correct_count = 0
            valid_accuracy = 0
            valid_loss = 0

            #--- Training Phase ---#
            model.train()    # set model to training mode

            pbar = tqdm(self.trainloader)
            pbar.set_description("Train")

            for x_batch, y_batch in pbar:      # take mini batch data from train_loader

                x_batch = x_batch.to(device)     # load x_batch data on GPU
                y_batch = y_batch.to(device)     # load y_batch data on GPU

                optimizer.zero_grad()           # reset gradients to 0
                p_batch = model(x_batch)        # do prediction
                loss = loss_func(p_batch, y_batch)   # measure loss
                loss.backward()              # calculate gradients
                optimizer.step()             # update model parameters

                train_loss += loss.item()                  # accumulate loss value
                # convert p_batch vector to p_batch_label
                p_batch_label = torch.argmax(p_batch, dim=1)
                # count up number of correct predictions
                train_correct_count += (p_batch_label == y_batch).sum()

                pbar.set_postfix(
                    {"accuracy": f"{(p_batch_label == y_batch).sum()/len(x_batch):.4f}", "loss": f"{loss.item():.4f}"})
            #----------------------#

            #--- Evaluation Phase ---#
            with torch.no_grad():   # disable autograd for saving memory usage
                model.eval()        # set model to evaluation mode

                pbar = tqdm(self.testloader)
                pbar.set_description("Valid")

                for x_batch, y_batch in pbar:   # take mini batch data from test_loader

                    x_batch = x_batch.to(device)     # load x_batch data on GPU
                    y_batch = y_batch.to(device)     # load y_batch data on GPU

                    p_batch = model(x_batch)         # do prediction
                    loss = loss_func(p_batch, y_batch)    # measure loss

                    valid_loss += loss.item()                  # accumulate loss value
                    p_batch_label = torch.argmax(p_batch, dim=1)
                    # convert p_batch vector to p_batch_label
                    # count up number of correct predictions
                    valid_correct_count += (p_batch_label == y_batch).sum()

                    pbar.set_postfix(
                        {"accuracy": f"{(p_batch_label == y_batch).sum()/len(x_batch):.4f}", "loss": f"{loss.item():.4f}"})
            #------------------------#

            # determine accuracy for training data
            train_accuracy = train_correct_count/len(self.trainset)
            # determin accuracy for test data
            valid_accuracy = valid_correct_count/len(self.testset)
            # determin loss for training data
            train_loss = train_loss/len(self.trainloader)
            # determin loss for validation data
            valid_loss = valid_loss/len(self.testloader)

            # show and store metrics
            print(
                f"Train: Accuracy={train_accuracy:.3f} Loss={train_loss:.3f}, Valid: Accuracy={valid_accuracy:.3f} Loss={valid_loss:.3f}")
            train_accuracy_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            valid_accuracy_list.append(valid_accuracy)
            valid_loss_list.append(valid_loss)

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), 'VGG16_model.pth')
                valid_loss_min = valid_loss

            print("---------------------")

        plt.subplot(2, 1, 1)
        plt.plot(np.arange(epochs)+1, train_accuracy_list, label="train")
        plt.plot(np.arange(epochs)+1, valid_accuracy_list, label="valid")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(epochs)+1, train_loss_list, label="train")
        plt.plot(np.arange(epochs)+1, valid_loss_list, label="valid")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()

        plt.savefig("training_loss_and_accuracy.png", pad_inches=0.0)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
