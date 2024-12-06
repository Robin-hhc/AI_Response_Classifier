# Files Provided

1. **`lstm_model.ipynb`**: Main script responsible for training and testing the dataset. Checkpoints, checkpoints metrics, models, and other model-related meta-data are saved by this script.
2. **`lstm_prediction.ipynb`**: Prediction script. Uses the model and tokenizer generated and saved from `lstm_model.ipynb`.
3. **`AI_Human.csv`**: Dataset used for training/testing. Not uploaded to Github due to its size.
4. **`./model`**: Directory containing model information saved by `lstm_model.ipynb`. Including checkpoint metrics, model, and the tokenizer information. Used by `lstm_prediction.ipynb` for prediction. `lstm_model.pkl` (model file) not provided due to Github's 100MB size limitation.
5. **`./html_generated`**: Directory containing static HTML of the `lstm_model.ipynb` and `lstm_prediction.ipynb` scripts after a complete execution. Includes graphs in the `lstm_model.ipynb`.

# How to run

### Requirements

#### Training

- Python 3.9+
- GPU with CUDA support
- 50GB Storage Minimum

#### Prediction

- `./lstm_prediction.ipynb`
- `./model/lstm_model.pkl`
- `./model/vocab.json`

### Install Dependencies

On Windows:

```
cd lstm
python -m venv lstm_venv
./lstm_venv/Script/activate
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

> Note: You must download the dataset `./AI_Human.csv` and place it in `lstm/`.
>
> If you would like to see the training process but do not want to re-run the training, you may find the static HTML results from the previous run under the `./html_generated` folder.

Run `lstm_model.ipynb` with the following constants to use the existing tokenizer and checkpoint data (recommended if these data has already been generated):

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

Under the `lstm/` folder, there is a notebook script called `lstm_prediction.ipynb` for running the predictions using the model downloaded above and the tokenizers provided.

The existing notebook output demostrates the prediction of a sentence in the last code-block. To predict something else, modify the `text` variable in the last code-block and re-run the notebook file.

If you are missing any required files (model and tokenizers), code-block 2 will output a warning, specifying the missing file.
