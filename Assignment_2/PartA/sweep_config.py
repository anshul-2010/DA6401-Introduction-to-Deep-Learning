# Hyperparameters for sweep

def sweep_configuration():
    sweep_config = {
        'method': 'bayes',
        'metric': {
        'name': 'model_accuracy',
        'goal': 'maximize'   
        },
        'parameters': {
            'epoch': {
                'values': [50]
            },
            'dropout':{
                'values': [0.1, 0.2, 0.3]
            },
            'number_of_filters': {
                'values': [32]
            },
            'filter_organisation': {
                'values': [2]
            },
            'size_filter':{
                'values': [3]
            },
            'data_augmentation': {
                'values':['Yes']
            },
            'batch_normalization': {
                'values':['Yes', 'No']
            },
            'learning_rate':{
                'values': [0.001, 0.01, 0.0001]
            },
            'optimizer_fn':{
                'values': ['adam', 'sgd', 'momentum']
            },
            'activation_fn':{
                'values': ['relu', 'tanh']
            },
            'dense':{
                'values': [128, 256, 512]
            },
            'stride':{
                'values': [2]
            }
        }
    }
    return sweep_config