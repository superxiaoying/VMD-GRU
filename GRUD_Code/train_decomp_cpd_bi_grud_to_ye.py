# -*- coding:utf8 -*-

import os
import glob

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
import pandas as pd
import keras
sys.stderr = stderr
from keras.layers import Bidirectional, Masking, Lambda, GRU, Dense, Dropout, Add
from keras.optimizers import Adam
# from keras.utils import custom_object_scope
import time
from nn_utils.grud_layers import Bidirectional_for_GRUD, GRUD
from nn_utils.layers import ExternalMasking

from data_gen_decomp_cpd import *

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


if __name__ == '__main__':

    # 所使用的超参数，需要根据显卡来修改
    params = {'batch_size': 10, # batch_size * n_samples batch size for training, 12 for RTX2070
            'dim': 63,
            'n_classes': 3,
            'n_samples': 300,
            'n_channels': 1,
            'shuffle': True}

    # 9种仿真数据，分别对应相应文件夹下的.npz文件
    sample_class = ['slowdn_linear_speedup', 'linear_slowdn', 'linear_speedup', 'speedup_linear_slowdn', 
                    'slowdn_linear','speedup_linear', 'linear', 'speedup', 'slowdn']
    
    # 读入train和val的文件列表
    train_list = "../data_cpd/train_d63_n5_m85_s15/{}/00*.npz"
    valid_list = "../data_cpd/valid_d63_n5_m85_s15/{}/00*.npz"

    filelist_train = []
    filelist_val = []
    for sample_class_sel in sample_class:
        # 文件的地址用list+文本的形式读入
        train_list_format = train_list.format(sample_class_sel)
        # glob.glob()通配符，把00*.npz相关的文件都获取
        filelist_train.extend(sorted(glob.glob(train_list_format)))
        print(train_list_format)
        val_list_format = valid_list.format(sample_class_sel)
        filelist_val.extend(sorted(glob.glob(val_list_format)))
        print(val_list_format)
    
    # 通过长度验证读取的数据是否写入正确
    print("train_list len: ", len(filelist_train))
    print("val_list len: ", len(filelist_val))

    # **代表写入参数的字典，DataGenerator是data_gen_decomp_cpd.py中的函数
    training_generator = DataGenerator(filelist_train, **params)
    validation_generator = DataGenerator(filelist_val, **params)

    # 生成一个run_id用于标识训练的模型
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    print("run id: ", run_id)

    # 超参数赋给变量
    input_dim = params['dim']
    input_ch = params['n_channels']
    input_noise = keras.layers.Input(shape=(None, input_ch))
    input_mask = keras.layers.Input(shape=(None, input_ch))
    input_stamp = keras.layers.Input(shape=(None, 1))

    # ExternalMasking 是GRU-D中的，从nn_utils.layers调用，用来表示数据缺失，An extension of `Masking` layer
    input_noise_masked = ExternalMasking()([input_noise, input_mask])
    input_stamp_masked = ExternalMasking()([input_stamp, input_mask])

    # Masking()是keras.layers中的
    input_mask_masked = Masking()(input_mask)

    # GRUD()、Bidirectional_for_GRUD()、从nn_utils.grud_layers中调用
    grud_layer = GRUD(units=64,  # 32 -> 64 -> 128
                      return_sequences=True,
                      activation='tanh',
                    #   dropout=0.1,
                      recurrent_dropout=0.3, # 0.3
                      x_imputation='zero',
                      input_decay='exp_relu',
                      hidden_decay='exp_relu',
                      feed_masking=True
                     )
    grud_layer = Bidirectional_for_GRUD(grud_layer)
    input_x = grud_layer([input_noise_masked, input_mask_masked, input_stamp_masked])
    input_x_cp = Lambda(lambda x: x, output_shape=lambda s:s)(input_x)

    lstm_1 = Bidirectional(GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True))(input_x_cp) # output shape (batch_size, time_steps, units)
    lstm_2 = Bidirectional(GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True))(lstm_1) # output shape (batch_size, time_steps, units) 64 - 32

    lstm_3r = Bidirectional(GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True))(lstm_2)  # 32 -16
    dense_2r = Dense(units=32, activation="tanh")(lstm_3r) # output shape (batch_size, time_steps, units) 8 -> 32
    dense_2r_drop = Dropout(rate=0.1)(dense_2r)
    dense_3r = Dense(units=1)(dense_2r_drop) # output shape (batch_size, time_steps, 1)
    output_seasonal = Lambda(lambda x: x * 200, name='season')(dense_3r) # magnify the output for loss balance

    lstm_3rs = Bidirectional(GRU(units=16, return_sequences=True, dropout=0, recurrent_dropout=0, reset_after=True))(lstm_2)  # 32 -16
    dense_2rs = Dense(units=32, activation="relu")(lstm_3rs) # output shape (batch_size, time_steps, units) 8 -> 32
    dense_2rs_drop = Dropout(rate=0.1)(dense_2rs)                     
    dense_3rs = Dense(units=1)(dense_2rs_drop) # output shape (batch_size, time_steps, 1)
    output_trend = Lambda(lambda x: x * 200, name='trend')(dense_3rs) # magnify the output for loss balance
    
    dense_2c = keras.layers.Dense(units=32, activation="relu")(lstm_3rs)
    dense_2c_drop = Dropout(rate=0.1)(dense_2c)                     
    output_label = keras.layers.Dense(units=params['n_classes'], activation='softmax', name='softmax')(dense_2c_drop) # softmax activation for classification output layer
    
    output_add = Add(name='add')([output_trend, output_seasonal])
    
    model = keras.models.Model(inputs=[input_noise, input_mask, input_stamp], outputs=[output_trend, output_seasonal, output_add, output_label]) #, input_mask, input_stamp

    opt = Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5) 
    
    loss_list = ['mse', 'mse', 'mse', 'categorical_crossentropy']
    metric_list = [[], [], [], ['accuracy']]  
    model.compile(loss=loss_list, loss_weights=[1, 1, 0, 10], metrics=metric_list, optimizer=opt)

    checkpoint_cb = keras.callbacks.ModelCheckpoint("./model_save/check_point/{run_id}.h5".format(run_id=run_id),  # source code adjusted
                                                    monitor='val_add_loss', verbose=1, save_best_only=True) #.{epoch:03d}-{val_loss:.4f}  
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor='val_add_loss', patience=5, verbose=1, restore_best_weights=True)  # source code adjusted
    # restore model weights from the epoch with the best value of the monitored quantity, verbose=1 to print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    model.load_weights("./model_select/model_mask_run_63.h5") # load pretrained-model and fintune

    history = model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                verbose=1,
                                use_multiprocessing=True,
                                workers=2,
                                epochs=3,
                                callbacks=[checkpoint_cb, early_stopping_cb]) 

    model_path = "./model_save/model_mask_{}.h5".format(run_id)
    weight_path = "./model_save/weights_{}.ckpt".format(run_id)
    model.save(model_path)
    print("Saving model", model_path)
    model.save_weights(weight_path)
    print("Saving weight", weight_path)

    # convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # or save to csv: 
    hist_csv_file = './learning_curves/lr_history_{}.csv'.format(run_id)
    with open(hist_csv_file, mode='w') as file_csv:
        hist_df.to_csv(file_csv)
        print("Saving history", hist_csv_file)
    
    hist_df.plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 40) # limit the y axis range
    learning_curve_file = "./learning_curves/lr_curves_{}".format(run_id)
    save_fig(learning_curve_file)
    print("Saving learning curve", learning_curve_file)
    plt.close()