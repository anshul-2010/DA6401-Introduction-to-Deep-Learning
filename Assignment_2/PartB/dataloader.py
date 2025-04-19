# Data Preparation with PyTorch DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def data_loader(data_path, data_type, batch_size=32, val_split=0.2):
    """
    Function to load data using PyTorch DataLoader.
    
    Parameters:
    - data_path: Path to the dataset.
    - data_type: Type of data ('train' or 'valid').
    - batch_size: Size of each batch.
    - val_split: Fraction of data to be used for validation.
    
    Returns:
    - train_loader/test_loader: DataLoader for training data/testing data.
    - valid_loader/extra_loader: DataLoader for validationdata/extra data.
    """
    
    img_height = 224
    img_width = 224
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_valid = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    if data_type == 'train':
        train_dataset = datasets.ImageFolder(root=data_path, transform=transform_train)
        valid_dataset = datasets.ImageFolder(root=data_path, transform=transform_valid)
        
        train_size = int((1 - val_split) * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_subset, valid_subset = random_split(train_dataset, [train_size, valid_size])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False)
        
        return train_loader, valid_loader
    
    elif data_type == 'test':
        test_dataset = datasets.ImageFolder(root=data_path, transform=transform_valid)
        extra_dataset = datasets.ImageFolder(root=data_path, transform=transform_valid)
        
        test_size = int(len(train_dataset))
        extra_size = len(train_dataset) - train_size
        test_subset, extra_subset = random_split(test_dataset, [test_size, extra_size])

        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        extra_loader = DataLoader(extra_subset, batch_size=batch_size, shuffle=False)
        
        return test_loader, extra_loader