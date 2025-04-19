# Hyperparameters for sweep

def sweep_configuration():
    sweep_config = {
        'method': 'bayes',
        'metric': {
        'name': 'model_accuracy',
        'goal': 'maximize'   
        },
        'parameters': {
            'base': {
                'values': ['resnet', 'inception', 'mobilenet', 'inceptionresnet']
            },
            'in_epoch': {
                'values': [3, 5, 10]
            },
            'ft_epoch': {
                'values': [4, 6, 10]
            },
            'ft_bool': {
                'values': ['Yes','No']
            },
            'dropout': {
                'values': [0.1,0.2]
            },
            'learning_rate': {
                'values': [0.001,0.0001]
            },
            'optimizer_fn': {
                'values': ['Adam']
            },
            'activation_fn': {
                'values': ['relu','tanh']
            },
            'dense': {
                'values': [256,512]
            }
        }
    }
    return sweep_config