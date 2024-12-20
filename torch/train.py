import os
import glob
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import pandas as pd

from nn_utils.grud_layers import Bidirectional_for_GRUD, GRUD
from nn_utils.layers import ExternalMasking
from data_gen_decomp_cpd import *

import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

params = {
    'batch_size': 10,
    'dim': 63,
    'n_classes': 3,
    'n_samples': 300,
    'n_channels': 1,
    'shuffle': True
}

sample_class = ['slowdn_linear_speedup', 'linear_slowdn', 'linear_speedup', 'speedup_linear_slowdn', 
                'slowdn_linear','speedup_linear', 'linear', 'speedup', 'slowdn']

train_list = "../data_cpd/train_d63_n5_m85_s15/{}/00*.npz"
valid_list = "../data_cpd/valid_d63_n5_m85_s15/{}/00*.npz"

filelist_train = []
filelist_val = []

for sample_class_sel in sample_class:
    train_list_format = train_list.format(sample_class_sel)
    filelist_train.extend(sorted(glob.glob(train_list_format)))
    print(train_list_format)
    
    val_list_format = valid_list.format(sample_class_sel)
    filelist_val.extend(sorted(glob.glob(val_list_format)))
    print(val_list_format)

print("train_list len: ", len(filelist_train))
print("val_list len: ", len(filelist_val))

training_generator = DataGenerator(filelist_train, **params)
validation_generator = DataGenerator(filelist_val, **params)

run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
print("run id: ", run_id)

input_dim = params['dim']
input_ch = params['n_channels']

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.externalMasking1 = ExternalMasking()
        self.externalMasking2 = ExternalMasking()
        self.grud_layer = Bidirectional_for_GRUD(GRUD(units=64, return_sequences=True, activation='tanh', recurrent_dropout=0.3, x_imputation='zero', input_decay='exp_relu', hidden_decay='exp_relu', feed_masking=True))
        self.bilstm1 = nn.GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True)
        self.bilstm2 = nn.GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True)
        self.bilstm3r = nn.GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True)
        self.bilstm3rs = nn.GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True)
        self.dense_2r = nn.Linear(16, 32)
        self.dropout_2r = nn.Dropout(0.1)
        self.dense_3r = nn.Linear(32, 1)
        self.dense_2rs = nn.Linear(16, 32)
        self.dropout_2rs = nn.Dropout(0.1)
        self.dense_3rs = nn.Linear(32, 1)
        self.dense_2c = nn.Linear(16, 32)
        self.dropout_2c = nn.Dropout(0.1)
        self.dense_3c = nn.Linear(32, params['n_classes'])

    def forward(self, input_noise, input_mask, input_stamp):
        input_noise_masked = self.externalMasking1(input_noise, input_mask)
        input_stamp_masked = self.externalMasking2(input_stamp, input_mask)
        input_x = self.grud_layer(input_noise_masked, input_mask, input_stamp_masked)
        input_x_cp = input_x
        lstm_1 = self.bilstm1(input_x_cp)
        lstm_2 = self.bilstm2(lstm_1)
        lstm_3r = self.bilstm3r(lstm_2)
        dense_2r = F.tanh(self.dense_2r(lstm_3r))
        dense_2r_drop = self.dropout_2r(dense_2r)
        dense_3r = self.dense_3r(dense_2r_drop)
        output_seasonal = dense_3r * 200
        lstm_3rs = self.bilstm3rs(lstm_2)
        dense_2rs = F.relu(self.dense_2rs(lstm_3rs))
        dense_2rs_drop = self.dropout_2rs(dense_2rs)
        dense_3rs = self.dense_3rs(dense_2rs_drop)
        output_trend = dense_3rs * 200
        dense_2c = F.relu(self.dense_2c(lstm_3rs))
        dense_2c_drop = self.dropout_2c(dense_2c)
        output_label = F.softmax(self.dense_3c(dense_2c_drop), dim=1)
        output_add = output_trend + output_seasonal
        return output_trend, output_seasonal, output_add, output_label

model = MyModel()

opt = optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)

# You need to define your own loss function here based on your requirements.
def custom_loss(output, target):
    pass

checkpoint_path = "./model_save/check_point/{run_id}.pth".format(run_id=run_id)
best_loss = float('inf')

# load pretrained-model and fine-tune
model.load_state_dict(torch.load("./model_select/model_mask_run_63.pth"))

for epoch in range(3):
    for inputs, targets in training_generator:
        opt.zero_grad()
        outputs = model(*inputs)
        loss = custom_loss(outputs, targets)
        loss.backward()
        opt.step()
    
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in validation_generator:
            outputs = model(*inputs)
            loss = custom_loss(outputs, targets)
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Validation loss: {val_loss}')
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saving model with validation loss {val_loss} at {checkpoint_path}')

# Save the trained model and weights
model_path = "./model_save/model_mask_{}.pth".format(run_id)
torch.save(model.state_dict(), model_path)
print("Saving model", model_path)

# Save the learning curve data
history = pd.DataFrame.from_dict(training_generator.history)
hist_csv_file = './learning_curves/lr_history_{}.csv'.format(run_id)
with open(hist_csv_file, mode='w') as file_csv:
    history.to_csv(file_csv)
    print("Saving history", hist_csv_file)

history.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 40) # limit the y axis range
learning_curve_file = "./learning_curves/lr_curves_{}.png".format(run_id)
plt.savefig(learning_curve_file)
print("Saving learning curve", learning_curve_file)
plt.close()