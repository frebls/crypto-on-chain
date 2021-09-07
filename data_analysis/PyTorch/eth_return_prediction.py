# -*- coding: utf-8 -*-
"""ETH_return_prediction

Ether market movement prediction with recurrent neural networks

"""

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
from collections import Counter

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

"""## Set config"""
sns.set_theme()

HAPPY_COLORS_PALETTE = ['#00388F', '#FFB400', '#FF4B00', '#65B800', '#00B1EA']

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 8, 5

tqdm.pandas()

pl.seed_everything(42)

"""## Load data"""

# from google.colab import auth

# auth.authenticate_user()
# print('Authenticated')

# %%bigquery df_ --project fiery-rarity-322109

# SELECT  TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(time), HOUR) as date, close, high, low, volumeto as volume
# FROM `fiery-rarity-322109.ethereum.eth_usd_min`
# where TIMESTAMP_SECONDS(time) = TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(time), HOUR) and DATE(TIMESTAMP_SECONDS(time)) <= '2021-07-30'
# order by date

df_ = pd.read_csv('ETH_features_hour.csv')

df_.head()

df_.shape

df = df_[5000:35000].copy() #[['date','close','high','low','volume']]
df = df.sort_values(by="date").reset_index(drop=True)
for col in df.columns:
  if col not in ['date','close']:
    df[col + '_chng'] = df[col].pct_change()
df.dropna(inplace=True)
df.head()

"""## Preprocessing"""

df["prev_close"] = df["close"].shift(1)

df["close_change"] = df.progress_apply(
    lambda row: 0 if np.isnan(row.prev_close) else np.log(row.close / row.prev_close),
    axis=1
)

df["close_change_encoded"] = df.progress_apply(
    lambda row: 0 if row.close_change <= 0 else 1,
    axis=1
)

df.head()

df.close.plot()
# df.close.plot()
# df.close_change.plot.hist(bins = 60)

df.columns

features_df = df[['close', 'close_change', 'close_change_encoded']].copy() #, 'to_exchange_chng', 'nr_addresses_chng', 'gini_chng', 'avg_balance_usd_chng'

train_size = int(len(features_df) * .8)
val_thr = int(len(features_df) * .9)
train_df, val_df, test_df = features_df[:train_size], features_df[train_size:val_thr], features_df[val_thr:]
train_df.shape, val_df.shape, test_df.shape

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_df)

train_df = pd.DataFrame( 
    scaler.transform(train_df), 
    index=train_df.index, 
    columns=train_df.columns
)

val_df = pd.DataFrame( 
    scaler.transform(val_df), 
    index=val_df.index, 
    columns=val_df.columns
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
    T = i + sequence_length
    sequence = input_data[i:T]
    label = 1 if input_data.iloc[T][target_column] > 0 else 0 # 1 if target_column[T] > 0 else 0
    sequences.append((sequence, label))

  return sequences

SEQUENCE_LENGTH = 24
TARGET_COLUMN = "close_change_encoded"

# train_sequences = create_sequences(train_df.loc[:, train_df.columns != TARGET_COLUMN].reset_index(drop=True), train_df[TARGET_COLUMN].reset_index(drop=True), SEQUENCE_LENGTH)
# val_sequences = create_sequences(val_df.loc[:, val_df.columns != TARGET_COLUMN].reset_index(drop=True), val_df[TARGET_COLUMN].reset_index(drop=True), SEQUENCE_LENGTH)
# test_sequences = create_sequences(test_df.loc[:, test_df.columns != TARGET_COLUMN].reset_index(drop=True), test_df[TARGET_COLUMN].reset_index(drop=True), SEQUENCE_LENGTH)

train_sequences = create_sequences(train_df, TARGET_COLUMN, SEQUENCE_LENGTH)
val_sequences = create_sequences(val_df, TARGET_COLUMN, SEQUENCE_LENGTH)
test_sequences = create_sequences(test_df, TARGET_COLUMN, SEQUENCE_LENGTH)

test_sequences[0][0].head()

len(train_sequences), len(val_sequences), len(test_sequences)

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
      self, train_sequences, val_sequences, test_sequences, batch_size=8
  ):
    super().__init__()
    self.train_sequences = train_sequences
    self.val_sequences = val_sequences
    self.test_sequences = test_sequences
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train_dataset = BTCDataset(self.train_sequences)
    self.val_dataset = BTCDataset(self.val_sequences)
    self.test_dataset = BTCDataset(self.test_sequences)

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=cpu_count()
    )
    
  def val_dataloader(self):
    return DataLoader(
        self.val_dataset,
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

N_EPOCHS = 20

config = {
  "LAYER_N": tune.choice([1, 2, 3]),
  "LAYER_S": tune.choice([64, 128, 256]),
  "LR": tune.loguniform(1e-4, 1e-1),
  "BATCH_SIZE": tune.choice([32, 64, 128]),
  "DROPOUT": tune.choice([0.2, 0.5, 0.8])
}

# BATCH_SIZE = 12
# LR = 0.0001
# LAYER_S = 256
# LAYER_N = 3
# DROPOUT = 0.8

n_features = train_sequences[0][0].shape[1]

train_dataset = BTCDataset(train_sequences)

for item in train_dataset:
  print(item["sequence"].shape)
  print(item["label"])
  break

"""## Model"""

class PricePredictionModel(nn.Module):

  def __init__(self, n_features, n_hidden=128, n_layers=2, dropout=0.2):
    super().__init__()

    self.n_hidden = n_hidden

    self.lstm = nn.LSTM(
        input_size=n_features,
        hidden_size=n_hidden,
        batch_first=True,
        num_layers=n_layers,
        dropout=dropout
    )

    self.classifier = nn.Linear(n_hidden, 1)
    # self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    self.lstm.flatten_parameters()

    _, (hidden, _) = self.lstm(x)
    out = hidden[-1]

    return self.classifier(out) # self.sigmoid()

class BTCPricePredictor(pl.LightningModule):

  def __init__(self, n_features: int, config: dict):
    super().__init__()

    self.n_layers = config["LAYER_N"]
    self.n_hidden = config["LAYER_S"]
    self.lr = config["LR"]
    self.dropout = config["DROPOUT"]

    self.model = PricePredictionModel(n_features, n_hidden, n_layers, dropout)
    self.criterion = nn.BCEWithLogitsLoss() #BCELoss

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
    predictions = (torch.sigmoid(outputs) > 0.5).long() #torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def validation_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    predictions = (torch.sigmoid(outputs) > 0.5).long() #torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def test_step(self, batch, batch_idx):
    sequences = batch["sequence"]
    labels = batch["label"]
    loss, outputs = self(sequences, labels)
    predictions = (torch.sigmoid(outputs) > 0.5).long() #torch.argmax(outputs, dim=1)
    step_accuracy = accuracy(predictions, labels)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    self.log("test_accuracy", step_accuracy, prog_bar=True, logger=True)
    return {"loss": loss, "accuracy": step_accuracy}

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)

model = BTCPricePredictor(n_features=n_features, lr=LR, n_hidden=LAYER_S, n_layers=LAYER_N, dropout=DROPOUT)

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

logger = TensorBoardLogger("lightning_logs", name=f"eth-lr {config["LR"]}-bs {config["BATCH_SIZE"]}-ls {config["LAYER_S"]}-ld {config["LAYER_N"]}-do {config["DROPOUT"]}")

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3)

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    gpus=torch.cuda.device_count(),
    progress_bar_refresh_rate=30
)

from ray.tune.integration.pytorch_lightning import TuneReportCallback

tune_callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "mean_accuracy": "val_accuracy"
    },
    on="validation_end")

callbacks=[early_stopping_callback, tune_callback]

def train_tune(config, callbacks, data_module, epochs=20):
  model = BTCPricePredictor(config)
  trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    max_epochs=epochs,
    gpus=torch.cuda.device_count(),
    progress_bar_refresh_rate=30,
    callbacks=[early_stopping_callback, tune_callback])
  trainer.fit(model)

data_module = BTCPriceDataModule(train_sequences, val_sequences, test_sequences, batch_size=config["BATCH_SIZE"])
train_tune(config, callbacks, data_module, epochs=N_EPOCHS)

data_module = BTCPriceDataModule(train_sequences, val_sequences, test_sequences, batch_size=config["BATCH_SIZE"][0])
trainer.fit(model, data_module)

trainer.test()

"""## Predictions"""

trainer.checkpoint_callback.best_model_path

trained_model = BTCPricePredictor.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path, #"best-checkpoint.ckpt",
    n_features=n_features,
    lr=LR, n_hidden=LAYER_S, n_layers=LAYER_N, dropout=DROPOUT
)

trained_model.freeze()

test_dataset = BTCDataset(test_sequences)

predictions = []
labels = []

for item in tqdm(test_dataset):
  sequence = item["sequence"]
  label = item["label"]

  _, output = trained_model(sequence.unsqueeze(dim=0))
  prediction = (torch.sigmoid(output) > 0.5).long() #torch.argmax(output, dim=1)
  predictions.append(prediction.item())
  labels.append(label.item())

len(predictions) == (len(test_df) - SEQUENCE_LENGTH)

from collections import Counter
Counter(predictions)

print(classification_report(labels, predictions)) #, target_names=

