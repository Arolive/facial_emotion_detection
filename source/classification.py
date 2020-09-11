##--- basic imports ---

import time
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

##--- Pytorch imports ----

import torch
# nn
import torch.nn as nn
import torch.nn.functional as F
# optim
import torch.optim as optim
from torch.optim import lr_scheduler
# utils
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
# torchvision
import torchvision
from torchvision import datasets, models, transforms

##--- Other imports ---

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


##--- Lib ---

cls2id = {"Happy": 0, "Sad": 1, "Fear": 2}
id2cls = ["Happy", "Sad", "Fear"]

BATCHSIZE = 10
#PATH = "../data/aithon2020_level2_traning.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



##--- Utility functions ---

def load_data(PATH):
    data     = pd.read_csv(PATH)
    try:
        labels   = data["emotion"]
        data     = data.drop(["emotion"], axis = 1)
    except:
        labels = None
    images   = np.array(data.values).reshape(len(data.values), 48, 48)
    images   = images/255
    return images, labels

def loader(PATH):
    images, labels = load_data(PATH)
    images = torch.tensor(images)
    images = images.view(images.shape[0], -1, images.shape[1], images.shape[2])

    if labels is not None:
        target = []
        for label in labels.values:
            target.append(cls2id[label])
        target = torch.tensor(target)
    else:
        target = None

    return images, target

def data_split(X, Y, test_size, shuffle = True):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, shuffle = shuffle)
    return(X_train, X_test, Y_train, Y_test)

def create_batch(X, Y, batch_size = 1):
    batch_x = [X[i: i + batch_size] for i in range(0, len(X), batch_size)]
    batch_y = [Y[i: i + batch_size] for i in range(0, len(Y), batch_size)]
    return list(zip(batch_x, batch_y))


##--- RESNET Model ---

class RESNET(nn.Module):
    def __init__(self, criterion = None, optimizer = None, learning_rate = 0.001, image_dimention = 1, categories = 3):
        super(RESNET, self).__init__()
        ## Defining networt
         # Defaulf input image dimention is 1
         # Default output categories is 3
        self.pretrained = models.resnet101(pretrained = True)
        self.pretrained.conv1 = nn.Conv2d(image_dimention, 64, kernel_size = (3, 3), stride=(2,2), padding=(3,3), bias=False)
        num_ftrs = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(num_ftrs, categories)

        ## Defining optimizer and loss function
         # Default loss function is cross entropy
         # Default optimizer is SGD
         # Default learning rate is 0.001
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(self.pretrained.parameters(), lr = learning_rate, momentum = 0.9)

    def forward(self, x):
        x = self.pretrained.forward(x)
        return x

    def train(self, traindata, valdata = None, numberEpoch = 10, DEBUG = True):
        trainlen = sum(list(batch[0].shape[0] for batch in traindata))
        total_batch = len(traindata)
        ## Loop over the dataset multiple times
        for epoch in range(numberEpoch):
            running_corrects = 0.0
            running_loss     = 0.0
            if DEBUG:
                pbar = tqdm(enumerate(traindata, 0), total = total_batch, desc = "Loss 0, Completed", ncols = 800)
            if not DEBUG:
                pbar = enumerate(traindata, 0)
            for count, data in pbar:
                inputs, labels = data[0].to(device), data[1].to(device)
                batch  = inputs.shape[0]
                inputs = inputs.type(torch.cuda.FloatTensor)

                ## zero the parameter gradients
                self.optimizer.zero_grad()

                ## forward + backward + optimize
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                ## Calculating statistics
                running_loss += loss.item() * batch
                running_corrects += torch.sum(preds == labels.data)

                ## Showing statistics
                if DEBUG:
                    pbar.set_description("Loss %.3f, Completed" %(running_loss/trainlen))
            if DEBUG:
                epoch_loss = running_loss/trainlen
                epoch_acc  = running_corrects/trainlen
                print('Epoch %d completed, average loss: %.3f, accuracy: %.3f' %(epoch + 1, epoch_loss, epoch_acc))

                if valdata:
                    val_loss, val_acc = self.evaluate(valdata)
                    print('Validation, average loss: %.3f, accuracy: %.3f' %(val_loss, val_acc))

    def evaluate(self, testdata):
        running_corrects = 0.0
        running_loss     = 0.0
        testlen = sum(list(batch[0].shape[0] for batch in testdata))
        for data in testdata:
            inputs, labels = data[0].to(device), data[1].to(device)
            batch  = inputs.shape[0]
            inputs = inputs.type(torch.cuda.FloatTensor)
            ## Forward
            outputs = self.forward(inputs)
            _, preds = torch.max(outputs, 1)
            ## Loss and accuracy
            loss = self.criterion(outputs, labels)
            running_loss += loss.item() * batch
            running_corrects += torch.sum(preds == labels.data)

        loss = running_loss/testlen
        acc  = running_corrects/testlen
        return loss, acc

    def predict(self, testdata, ID = None):
        predicted_labels = []
        for data in testdata:
            inputs, labels = data[0].to(device), data[1].to(device)
            batch  = inputs.shape[0]
            inputs = inputs.type(torch.cuda.FloatTensor)
            ## Forward
            outputs = self.forward(inputs)
            _, preds = torch.max(outputs, 1)
            predicted_labels += preds.tolist()
        if ID:
            return([ID[label] for label in predicted_labels])
        return predicted_labels


##--- VGG Model ---

class VGGNET(nn.Module):
    def __init__(self, criterion = None, optimizer = None, learning_rate = 0.001, image_dimention = 1, categories = 3):
        super(VGGNET, self).__init__()
        ## Defining networt
         # Defaulf input image dimention is 1
         # Default output categories is 3
        self.pretrained = models.vgg19(pretrained = True)
        self.pretrained.features[0] = nn.Conv2d(image_dimention, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        num_ftrs = self.pretrained.classifier[6].in_features
        self.pretrained.classifier[6] = nn.Linear(num_ftrs, categories)

        ## Defining optimizer and loss function
         # Default loss function is cross entropy
         # Default optimizer is SGD
         # Default learning rate is 0.001
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(self.pretrained.parameters(), lr = learning_rate, momentum = 0.9)

    def forward(self, x):
        x = self.pretrained.forward(x)
        return x

    def train(self, traindata, valdata = None, numberEpoch = 10, DEBUG = True):

        trainlen = sum(list(batch[0].shape[0] for batch in traindata))
        total_batch = len(traindata)
        ## Loop over the dataset multiple times
        for epoch in range(numberEpoch):
            running_corrects = 0.0
            running_loss     = 0.0
            if DEBUG:
                pbar = tqdm(enumerate(traindata, 0), total = total_batch, desc = "Loss 0, Completed", ncols = 800)
            else:
                pbar = enumerate(traindata, 0)
            for count, data in pbar:
                inputs, labels = data[0].to(device), data[1].to(device)
                batch  = inputs.shape[0]
                inputs = inputs.type(torch.cuda.FloatTensor)

                ## zero the parameter gradients
                self.optimizer.zero_grad()

                ## forward + backward + optimize
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                ## Calculating statistics
                running_loss += loss.item() * batch
                running_corrects += torch.sum(preds == labels.data)

                ## Showing statistics
                if DEBUG:
                    pbar.set_description("Loss %.3f, Completed" %(running_loss/trainlen))
            if DEBUG:
                epoch_loss = running_loss/trainlen
                epoch_acc  = running_corrects/trainlen
                print('Epoch %d completed, average loss: %.3f, accuracy: %.3f' %(epoch + 1, epoch_loss, epoch_acc))

                if valdata:
                    val_loss, val_acc = self.evaluate(valdata)
                    print('Validation, average loss: %.3f, accuracy: %.3f' %(val_loss, val_acc))

    def evaluate(self, testdata):
        running_corrects = 0.0
        running_loss     = 0.0
        testlen = sum(list(batch[0].shape[0] for batch in testdata))
        with torch.no_grad():
            for data in testdata:
                inputs, labels = data[0].to(device), data[1].to(device)
                batch  = inputs.shape[0]
                inputs = inputs.type(torch.cuda.FloatTensor)
                ## Forward
                outputs = self.forward(inputs)
                _, preds = torch.max(outputs, 1)
                ## Loss and accuracy
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * batch
                running_corrects += torch.sum(preds == labels.data)

        loss = running_loss/testlen
        acc  = running_corrects/testlen
        return loss, acc

    def predict(self, testdata, ID = None):
        predicted_labels = []
        for data in testdata:
            inputs, labels = data[0].to(device), data[1].to(device)
            batch  = inputs.shape[0]
            inputs = inputs.type(torch.cuda.FloatTensor)
            ## Forward
            outputs = self.forward(inputs)
            _, preds = torch.max(outputs, 1)
            predicted_labels += preds.tolist()
        if ID:
            return([ID[label] for label in predicted_labels])
        return predicted_labels


##--- core functions ---

def train_a_model(trainfile):

    ## Loading images
    images, target = loader(trainfile)
    ## Train test split
    train_X, test_X, train_Y, test_Y = data_split(images, target, test_size = 0.3)
    ## Train loader
    trainloader = create_batch(train_X, train_Y, batch_size = BATCHSIZE)
    ## Test loader
    testloader = create_batch(test_X, test_Y, batch_size = BATCHSIZE)
    
    # Defining models
    model_XGB = XGBClassifier(max_depth = 3000)
    model_XGB.fit(train_X, train_Y)

    return model_XGB
  
def test_the_model(model, testfile):

    ## XGB
    prediction_xgb = model.predict(test_X)
    
    return prediction_xgb

