# Files Provided
1. **`lstm_model.ipynb`**: Main script responsible for training and testing the dataset. Checkpoints, checkpoints metrics, models, and other model-related meta-data are saved by this script.
2. **`lstm_prediction.ipynb`**: Prediction script. Uses the model and tokenizer generated and saved from `lstm_model.ipynb`.
3. **`AI_Human.csv`**: Dataset used for training/testing. Not uploaded to Github due to its size.
4. **`./model`**: Directory containing model information saved by `lstm_model.ipynb`. Including checkpoint metrics, model, and the tokenizer information. Used by `lstm_prediction.ipynb` for prediction.
5. **`./html_generated`**: Directory containing static HTML of the  `lstm_model.ipynb` and `lstm_prediction.ipynb` scripts after a complete execution. Includes graphs in the  `lstm_model.ipynb`.

# How to run

### Requirements

#### Training
 - Python 3.10+
 - GPU with CUDA support
 - 50GB Storage Minimum

 #### Prediction
  - `./model/lstm_model.pkl`
  - `./model/vocab.json`

### Install Dependencies

On Windows:
```
python -m venv venv
./venv/Script/activate
pip install -r requirements.txt
```

**For training the model**, depending on the CUDA version installed, you'll have to install the corresponding torch version.

For this project, we are using the CUDA v11.8:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```
and
```
torch==2.5.1+cu118
```

Note: It's **not recommended** to run `lstm_model.ipynb` on the CPU since the estimated time of completion is 10,000+ hours based on the current model settings.

### Run

#### Train Model
> Note: Since the dataset is not provided in the git repo, `lstm_model.ipynb` cannot be rerun. You may find the static HTML results from the previous run under the `./html_generated` folder.
> 
> Use the following guide is the dataset is available.

Run `lstm_model.ipynb` with the following arguments to use the existing tokenizer and checkpoint data (recommended if these data has already been generated):
```python
WRITE_VOCAB = False                   # Required for prediction
OVERWRITE_MODEL = True                # Required for prediction
SAVE_CHECKPOINTS = False
LOAD_CHECKPOINT_METRICS = True
WRITE_CHECKPOINT_METRICS = False
```
Run `lstm_model.ipynb` with the following arguments to generate new model, tokenizer, and checkpoints:
```python
WRITE_VOCAB = True                   # Required for prediction
OVERWRITE_MODEL = True               # Required for prediction
SAVE_CHECKPOINTS = True
LOAD_CHECKPOINT_METRICS = False
WRITE_CHECKPOINT_METRICS = True
```

Checkpoint files will be generated under `./checkpoints` based on the `save_steps` defined in the program. It's recommended to leave it as it is since lowering it below 1000 will drastically increase the runtime and the storage needed for the checkpoint files. 

#### Prediction
Run `lstm_prediction.ipynb` for prediction. Modify the last section to predict the text you could like to predict.