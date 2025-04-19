import torch
import torch
import wandb
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, num_epochs, ft_start=0, scheduler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders[0]
            else:
                model.eval()
                dataloader = dataloaders[1]
                
            running_loss, running_corrects = 0.0, 0

            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    # For InceptionV3, outputs are in InceptionOutputs, so we need to get the logits
                    if isinstance(outputs, tuple):  # Check if it's an InceptionOutputs tuple
                        outputs = outputs[0]  # Get the logits from the tuple
                    
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_acc": epoch_acc})
            history[f"{'' if phase == 'train' else 'val_'}loss"].append(epoch_loss)
            history[f"{'' if phase == 'train' else 'val_'}acc"].append(epoch_acc)

        if scheduler:
            scheduler.step()
    return history