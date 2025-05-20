import os
import random
import time
import wandb
import re, string
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from colour import Color
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.text import Tokenizer

from load_data import download_data
from preprocess_data import get_data_files, preprocess_data
from model import Seq2SeqModel
from visualize_connectivity import get_connectivity, get_lstm_output, get_output_from_embedding, visualise_connectivity
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
       
def get_test_words(n):
        test_df = pd.read_csv(get_data_files("hi")[2])
        test_sample = test_df.sample(n)
        test_sample.reset_index(inplace=True, drop=True)
        test_words = []
        for i in test_sample.index:
            entry = test_sample["अंक\tank\t5"].loc[i]
            parts = entry.split("\t")
            word = parts[1]
            test_words.append(word)
        return test_words
     
def __main__():
    download_data("/DakshinaDataset")

    # Call the testing phase function to test the model on the dataset without WandB
    model = test_on_dataset(language="hi", embedding_dim=256, encoder_layers=3, decoder_layers=3,
                            layer_type="lstm", units=256, dropout=0.2, attention=False)
    
    # Visualize the model outputs by adjusting the number of samples and number of words to visualize
    visualize_model_outputs(model, n=20)
    
    # Randomly evaluate the model on a few words
    # This function will randomly select a few words from the test set and evaluate the model on them
    randomly_evaluate(model, n=5)

    # Obtain the test words from the test set
    # This function will randomly select a few words from the test set and return them
    test_words = get_test_words(5)
    print(test_words)
    
    # Visualize the connectivity of the model for the test words
    # This function will visualize the connectivity of the model for the given words
    for word in test_words:
        visualise_connectivity(model, word, activation="scaler")