import time

import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, model_type=models.inception_v3(pretrained=True)):
        self.model = model_type
        self.lr = 0.01
        self.momentum = 0.9
        self.parameters = model_type.parameters()
        self.optimizer = optim.SGD(self.parameters, lr=self.lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataloader = None
        self.test_dataloader = None
        self.output_num = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_lr_momentum(self, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.parameters = self.model.parameters()
        self.optimizer = optim.SGD(self.parameters, lr=self.lr, momentum=self.momentum)

    def update_model(self, criterion, optimizer):
        self.criterion = criterion
        self.optimizer = optimizer

    def initialize_data(self, train_foldername, test_foldername, train_batch_size, test_batch_size, shuffle=True):
        train_data = datasets.ImageFolder(
            os.path.join(os.getcwd(), train_foldername),
            transform=transforms.Compose(
                [transforms.Resize((299, 299)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            ),
        )
        test_data = datasets.ImageFolder(
            os.path.join(os.getcwd(), test_foldername),
            transform=transforms.Compose(
                [transforms.Resize((299, 299)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            ),
        )
        self.output_num = len(train_data.classes)
        self.train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=shuffle)
        self.test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=shuffle)

    def train(self, epochs):
        since = time.time()
        num_epochs = np.arange(1, epochs+1, step=1)
        training_accuracy_plotter = []
        testing_accuracy_plotter = []
        for epoch in num_epochs:
            print('Epoch {}/{}'.format(epoch, epochs))
            print('+' * 10)
            self.model.train()
            running_loss = 0.0
            running_corrects = 0
            num_samples = 0

            for inputs, labels in self.train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, aux_outputs = self.model(inputs)
                loss1 = self.criterion(outputs, labels)
                loss2 = self.criterion(aux_outputs, labels)
                loss = 0.8 * loss1 + 0.2 * loss2
                loss.backward()
                self.optimizer.step()
                _, preds = torch.max(outputs, 1)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_samples += len(preds)

            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            epoch_acc = running_corrects / num_samples
            print("Loss: ", epoch_loss)
            print("Training Accuracies: ", epoch_acc)
            training_accuracy_plotter.append(epoch_acc)
            accuracy = self.test_model()
            testing_accuracy_plotter.append(accuracy)
            print("Testing Accuracy: ", accuracy)
        plt.plot(num_epochs, training_accuracy_plotter, label='Training Accuracy')
        plt.plot(num_epochs, testing_accuracy_plotter, label='Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Testing/Training Accuracies')
        plt.legend()
        time_elapsed = time.time() - since
        print()
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        plt.show()

    def test_model(self):
        self.model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                _, predictions = torch.max(self.model(inputs), 1)
                num_correct += torch.sum(predictions == labels.data)
                num_samples += len(predictions)
        return num_correct / num_samples

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
