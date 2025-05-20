import wandb
import tensorflow as tf

from load_data import download_data
from preprocess_data import get_data_files, preprocess_data
from model import Seq2SeqModel
from visualize_connectivity import get_connectivity, get_lstm_output, get_output_from_embedding
from visualize_output import visualize_model_outputs, randomly_evaluate
from testing_phase import test_on_dataset
from utils import BeamSearch, get_lstm_output, get_output_from_embedding, get_connectivity, get_colors, Colorizer

def train_with_wandb(language, test_beam_search=False):

    config_defaults = {"embedding_dim": 64, 
                       "enc_dec_layers": 1,
                       "layer_type": "lstm",
                       "units": 128,
                       "dropout": 0,
                       "attention": False,
                       "beam_width": 3,
                       "teacher_forcing_ratio": 1.0
                       }

    wandb.init(config=config_defaults, project="DA6401-Assignment-3", resume=True, entity="anshul_2010-indian-institute-of-technology-madras")
    # Below is an example of a custom run name for sweep 4
    # This line was different for all sweeps
    #wandb.run.name = f"beam_width_{wandb.config.beam_width}"

    ## 1. SELECT LANGUAGE ##
    TRAIN_TSV, VAL_TSV, TEST_TSV = get_data_files(language)

    ## 2. DATA PREPROCESSING ##
    dataset, input_tokenizer, targ_tokenizer = preprocess_data(TRAIN_TSV)
    val_dataset, _, _ = preprocess_data(VAL_TSV, input_tokenizer, targ_tokenizer)

    ## 3. CREATING THE MODEL ##
    model = Seq2SeqModel(embedding_dim=wandb.config.embedding_dim,
                         encoder_layers=wandb.config.enc_dec_layers,
                         decoder_layers=wandb.config.enc_dec_layers,
                         layer_type=wandb.config.layer_type,
                         units=wandb.config.units,
                         dropout=wandb.config.dropout,
                         attention=wandb.config.attention)
    
    ## 4. COMPILING THE MODEL 
    model.set_vocabulary(input_tokenizer, targ_tokenizer)
    model.build(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(),
                metric = tf.keras.metrics.SparseCategoricalAccuracy())
    
    ## 5. FITTING AND VALIDATING THE MODEL
    model.fit(dataset, val_dataset, epochs=30, use_wandb=True, teacher_forcing_ratio=wandb.config.teacher_forcing_ratio)

    if test_beam_search:
        ## OPTIONAL :- Evaluate the dataset using beam search and without beam search
        val_dataset, _, _ = preprocess_data(VAL_TSV, model.input_tokenizer, model.targ_tokenizer)
        subset = val_dataset.take(500)

        # a) Without beam search
        _, test_acc_without = model.evaluate(subset, batch_size=100) 
        wandb.log({"test acc": test_acc_without})
        
        # b) With beam search
        beam_search = BeamSearch(model=model, k=wandb.config.beam_width)
        beam_search.evaluate(subset, batch_size=100, use_wandb=True)
        
def __main__():
    download_data("/DakshinaDataset")

    wandb.login(key="Enter your wandb key here")
    
    # Sweep Configuration 2
    sweep_config2 = {
        "name": "Sweep 2- Assignment3",
        "description": "Hyperparameter sweep for Seq2Seq Model without Attention",
        "method": "grid",
        "metric": {'name': 'val_acc', 'goal': 'maximize'},
        "parameters": {
                "enc_dec_layers": {"values": [2, 3]},
                "embedding_dim": {"values": [32, 64, 128, 256]},
                "dropout": {"values": [0.2, 0.3, 0.4]}
                }
        }
    sweep_id = wandb.sweep(sweep_config2, project="Enter your project name here")
    wandb.agent(sweep_id, function=lambda: train_with_wandb("hi"))