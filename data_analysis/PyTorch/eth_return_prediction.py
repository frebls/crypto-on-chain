# -*- coding: utf-8 -*-
"""ETH_return_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Zv8Xh3p7-6ffOlDOvOZdRwd8juCyK8kK
"""

# !nvidia-smi

# !pip install --quiet pytorch-lightning==1.2.5

# !pip install --quiet tqdm==4.59.0

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import math
import matplotlib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm #notebook
from multiprocessing import cpu_count
# from collections import defaultdict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

# sns.set(style='whitegrid', palette='muted', font_scale=1.2)
sns.set_theme()

HAPPY_COLORS_PALETTE = ['#00388F', '#FFB400', '#FF4B00', '#65B800', '#00B1EA']

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 8

tqdm.pandas()

pl.seed_everything(42)

"""## Load data"""

from google.colab import auth

auth.authenticate_user()
print('Authenticated')

# Commented out IPython magic to ensure Python compatibility.
# %%bigquery df_ --project fiery-rarity-322109
# 
# SELECT  TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(time), HOUR) as date, close, high, low, volumeto as volume
# FROM `fiery-rarity-322109.ethereum.eth_usd_min`
# where TIMESTAMP_SECONDS(time) = TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(time), HOUR) and DATE(TIMESTAMP_SECONDS(time)) <= '2021-07-30'
# order by date

df = df_[-50000:].copy()
df = df.sort_values(by="date").reset_index(drop=True)
df.head()

"""## Preprocessing"""

df["prev_close"] = df.shift(1)["close"]

df["close_change"] = df.progress_apply(
    lambda row: 0 if np.isnan(row.prev_close) else np.log(row.close / row.prev_close),
    axis=1
)

df["close_change_encoded"] = df.progress_apply(
    lambda row: 0 if row.close_change <= 0 else 1,
    axis=1
)

df.head()

features_df = df.copy()
del features_df["date"]
del features_df["prev_close"]

train_size = int(len(features_df) * .9)
train_df, test_df = features_df[:train_size], features_df[train_size:] #TODO check whether +1 is to be removed
train_df.shape, test_df.shape

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_df)

train_df = pd.DataFrame( 
    scaler.transform(train_df), 
    index=train_df.index, 
    columns=train_df.columns
)

test_df = pd.DataFrame( 
    scaler.transform(test_df), 
    index=test_df.index, 
    columns=test_df.columns
)

train_df.head()

def create_sequences(input_data: pd.DataFrame, target_column, sequence_length):

  sequences = []
  data_size = len(input_data)

  for i in tqdm(range(data_size - sequence_length)):
    sequence = input_data[i:i+sequence_length]

    label_position = i + sequence_length
    label = 1 if input_data.iloc[label_position][target_column] > 0 else 0
    sequences.append((sequence, label))

  return sequences

SEQUENCE_LENGTH = 24

train_sequences = create_sequences(train_df, "close_change_encoded", SEQUENCE_LENGTH)
test_sequences = create_sequences(test_df, "close_change_encoded", SEQUENCE_LENGTH)

test_sequences[0][0].head()

len(train_sequences), len(test_sequences)

assert (train_size - len(train_sequences)) == SEQUENCE_LENGTH

"""## PyTorch Dataset"""

class BTCDataset(Dataset):

  def __init__(self, sequences):
    self.sequences = sequences

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, idx):
    sequence, label = self.sequences[idx]

    return dict(
        sequence=torch.Tensor(sequence.to_numpy()),
        label=torch.tensor(label).long()
    )

class BTCPriceDataModule(pl.LightningDataModule):

  def __init__(
      self, train_sequences, test_sequences, batch_size=8
  ):
    super().__init__()
    self.train_sequences = train_sequences
    self.test_sequences = test_sequences
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train_dataset = BTCDataset(self.train_sequences)
    self.test_dataset = BTCDataset(self.test_sequences)

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=cpu_count()
    )
    
  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cpu_count()
    )
    
  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cpu_count()
    )

N_EPOCHS = 8
BATCH_SIZE = 128
LR = 0.001

data_module = BTCPriceDataModule(train_sequences, test_sequences, batch_size=BATCH_SIZE)
# data_module.setup() # not needed

train_dataset = BTCDataset(train_sequences)

for item in train_dataset:
  print(item["sequence"].shape)
  # print(item["label"].shape)
  print(item["label"])
  break

"""## Model"""

class PricePredictionModel(nn.Module):

  def __init__(self, n_features, n_hidden=128, n_layers=2):
    super().__init__()

    self.n_hidden = n_hidden

    self.lstm = nn.LSTM(
        input_size=n_features,
        hidden_size=n_hidden,
        batch_first=True,
        num_layers=n_layers,
        dropout=0.2
    )

    self.classifier = nn.Linear(n_hidden, 1)

  def forward(self, x):
    self.lstm.flatten_parameters()

    _, (hidden, _) = self.lstm(x)
    out = hidden[-1]

    return self.classifier(out)

class BTCPricePredictor(pl.LightningModule):

  def __init__(self, n_features: int, lr: float):
    super().__init__()
    self.model = PricePredictionModel(n_features)
    self.criterion = nn.BCEWithLogitsLoss()
    self.lr = lr

  def forward(self, x, labels=None):
    output = self.model(x)
    loss = 0
    if labels is not None:
      # print(output, labels.unsqueeze(dim=1).type_as(output))
      loss = self.criterion(output, labels.unsqueeze(dim=1).type_as(output))
    return loss, output

  def training_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def validation_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def test_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    predictions = torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)

model = BTCPricePredictor(n_features=train_df.shape[1], lr=LR)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir ./lightning_logs

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="eth-return")

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    # gpus=1,
    progress_bar_refresh_rate=30
)

trainer.fit(model, data_module)

trainer.test()

"""## Predictions"""

trained_model = BTCPricePredictor.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path, #"/content/checkpoints/best-checkpoint.ckpt", #"best-checkpoint.ckpt",
    n_features=train_df.shape[1],
    lr=LR
)

trained_model.freeze()

test_dataset = BTCDataset(test_sequences)

predictions = []
labels = []

for item in tqdm(test_dataset):
  sequence = item["sequence"]
  label = item["label"]

  _, output = trained_model(sequence.unsqueeze(dim=0))
  prediction = torch.argmax(output, dim=1)
  predictions.append(prediction.item())
  labels.append(label.item())

len(predictions) == (len(test_df) - SEQUENCE_LENGTH)

print(classification_report(labels, predictions)) #, target_names=

