# Define the CNN Model in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Model(nn.Module):
    def __init__(self, num_filters, filter_organisation, size_filter, dropout, activation_fn, dense_units, stride, batch_normalization):
        super(CNN_Model, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=num_filters, 
                               kernel_size=size_filter, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters) if batch_normalization == 'Yes' else nn.Identity()
        self.conv2 = nn.Conv2d(in_channels=num_filters, 
                               out_channels=num_filters * filter_organisation, 
                               kernel_size=size_filter, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * filter_organisation) if batch_normalization == 'Yes' else nn.Identity()
        self.conv3 = nn.Conv2d(in_channels=num_filters * filter_organisation, 
                               out_channels=num_filters * filter_organisation * filter_organisation, 
                               kernel_size=size_filter, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters * filter_organisation * filter_organisation) if batch_normalization == 'Yes' else nn.Identity()
        self.conv4 = nn.Conv2d(in_channels=num_filters * filter_organisation * filter_organisation, 
                               out_channels=num_filters * filter_organisation * filter_organisation, 
                               kernel_size=size_filter, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_filters * filter_organisation * filter_organisation) if batch_normalization == 'Yes' else nn.Identity()
        self.conv5 = nn.Conv2d(in_channels=num_filters * filter_organisation * filter_organisation, 
                               out_channels=num_filters * filter_organisation * filter_organisation, 
                               kernel_size=size_filter, stride=stride, padding=1)
        self.bn5 = nn.BatchNorm2d(num_filters * filter_organisation * filter_organisation) if batch_normalization == 'Yes' else nn.Identity()
        
        self.fc1 = nn.Linear(num_filters * filter_organisation * filter_organisation * 4, dense_units)
        self.fc2 = nn.Linear(dense_units, 10)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation_fn == 'relu' else torch.tanh
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.activation(x)
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)