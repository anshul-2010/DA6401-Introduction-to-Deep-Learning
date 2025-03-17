import numpy as np
import wandb
from Utility.hyperparam_tune import *

sweep_config = {
  "name": "CS6910 Assignment 1 - Cross Entropy Loss",
  "metric": {
      "name":"validation_accuracy",
      "goal": "maximize"
  },
  "method": "random",
  "parameters": {
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        "activation_f": {
            "values": ["sigmoid", "relu", "tanh"]
        },
        "init_mode": {
            "values": ["xavier", "random_uniform", "random_normal"]
        },
        "optimizer": {
            "values": ["sgd", "momentum", "nestorov", "adam", "nadam", "RMSprop"]
        },
        "batch_size": {
            "values": [16,32,64,128,256,512]
        },
        "epochs": {
            "values": [5, 10, 20, 40]
        },
        "L2_lamb": {
            "values": [0, 0.0005, 0.5]
        },
        "num_neurons": {
            "values": [32, 64, 128]
        },
        "num_hidden": {
            "values": [3, 4, 5]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment1")
wandb.agent(sweep_id, NN_fit, count=120)