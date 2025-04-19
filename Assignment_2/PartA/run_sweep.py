import wandb
from train import train_model
from sweep_config import sweep_configuration

wandb.login()

# Sweep function
def train():
    config_defaults = {
        'epoch': 50,
        'dropout': 0.2,
        'number_of_filters': 32,
        'filter_organisation': 2,
        'size_filter': 3,
        'data_augmentation': 'Yes',
        'batch_normalization': 'Yes',
        'learning_rate': 0.001,
        'optimiser_fn': 'adam',
        'activation_fn': 'relu',
        'dense': 256,
        'stride': 2    
    }
    
    # Initialize a new wandb run
    wandb.init(config=config_defaults, resume='allow')
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    
    train_model(
        config.epoch,
        config.dropout,
        config.number_of_filters,
        config.filter_organisation,
        config.size_filter,
        config.data_augmentation,
        config.batch_normalization,
        config.learning_rate,
        config.optimiser_fn,
        config.activation_fn,
        config.dense,
        config.stride
    )

sweep_config = sweep_configuration()
sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment-2", entity="anshul_2010-indian-institute-of-technology-madras")

# Run the sweep agent
# This will run the train function with different hyperparameters 15 times
wandb.agent(sweep_id, train, count=15)