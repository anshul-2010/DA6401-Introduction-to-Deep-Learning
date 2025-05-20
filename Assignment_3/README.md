# DA6401-Assignment-3

1. This notebook is structured in such a way that all the cells can be run one after another. Run All Cells command can also be used, but be wary of WandB sweeps at the end.
2. To run the model without WandB, use the following code:
```python
model = test_on_dataset(language="hi",
                        embedding_dim=256,
                        encoder_layers=3,
                        decoder_layers=3,
                        layer_type="lstm",
                        units=256,
                        dropout=0.2,
                        attention=False)
```
Alternatively, you can also run the python script corresponding to the same:
```python
python main.py
```
3. To run the model with WandB sweep, use the following code:
```python
# Creating the WandB config
sweep_config = {
  "name": "Sweep 1- Assignment3",
  "method": "grid",
  "metric": {'name': 'val_acc', 'goal': 'maximize'},
  "parameters": {
        "enc_dec_layers": {
           "values": [1, 2, 3]
        },
        "units": {
            "values": [64, 128, 256]
        },
        "layer_type": {
            "values": ["rnn", "gru", "lstm"]
        }
    }
}
# Creating a sweep
sweep_id = wandb.sweep(sweep_config, project="Enter your project name here")
# Running the sweep
wandb.agent(sweep_id, function=lambda: train_with_wandb("hi"))
```
Alternatively, you can also run the python script corresponding to each of the sweeps:
```python
python Sweeps_main/main_sweep1.py
python Sweeps_main/main_sweep2.py
python Sweeps_main/main_sweep3.py
python Sweeps_main/main_sweep4.py
python Sweeps_main/main_sweep5.py
```
4. To visualize the model outputs in the form of a wordcloud, use the following code:
```python
# "model" here is the output of the function in step 2
visualize_model_outputs(model, n=20)
```
5. To visualise the model connectivity, use the following code:
```python
# Sample some words from the test data
test_words = get_test_words(5)
# Visualise connectivity for "test_words"
for word in test_words:
    visualise_connectivity(model, word, activation="scaler")
```
6. WandB report can be found at: https://wandb.ai/anshul_2010-indian-institute-of-technology-madras/DA6401-Assignment-3/reports/Assignment-3-Seq2Seq-Neural-Transliteration--VmlldzoxMjc1NTAyNA?accessToken=ao02rlxk9rf2gam86oov0lwyywuy6vm2aqighjtu5ve8z9kdhxfusw35xufllbfv
