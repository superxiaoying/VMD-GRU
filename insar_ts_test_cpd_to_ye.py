import os
import sys

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use id from $ nvidia-smi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
# from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
sys.stderr = stderr
from keras.utils import custom_object_scope
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import pandas as pd
import geopandas as gpd
from dateutil.parser import parse
from scipy.ndimage import gaussian_filter1d
from datetime import timedelta  
# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=10)
mpl.rc('xtick', labelsize=10)
mpl.rc('ytick', labelsize=10)
mpl.rc('font', weight='bold')

def save_fig(fig_path, tight_layout=True, fig_extension="png", resolution=300, transparent=False):
    path = os.path.join(fig_path + "." + fig_extension)
    # print("Saving figure", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, transparent=transparent)

def load_mask_model(file_name):
    from nn_utils import _get_scope_dict
    with custom_object_scope(_get_scope_dict()):
        model = keras.models.load_model(file_name)
    return model

def scale_normalize(data):
    scale = 2/110  # (1 - -1)/(30 - -80)  # changed
    data_scaled = scale * (data + 80) -1  # scale to (-1 - 1)
    return data_scaled

def gauss_smooth_missed(data_nan=None, filter_sigma=None):
    """ This method convolves the data with a gaussian
    the smoothed data is returned
    @param array data: raw data
    @param int filter_len: length of filter, window size
    @param int filter_sigma: width of gaussian
    @return array smoothed: smoothed data
    """
    # gauss_kern = gaussian(2*filter_len, std=filter_sigma)

    data_zero = data_nan.copy()
    data_zero[np.isnan(data_nan)] = 0

    mask = 0*data_nan.copy() + 1
    mask[np.isnan(data_nan)] = 0

    smoothed = gaussian_filter1d(data_zero, sigma=filter_sigma, axis=-1, mode='reflect', truncate=4)  # radius = int(truncate * sd + 0.5), window = 2*radius
    smoothed_mask = gaussian_filter1d(mask, sigma=filter_sigma, axis=-1, mode='reflect', truncate=4)

    smoothed_mask[smoothed_mask==0] = 1
    smoothed = smoothed/smoothed_mask
    return smoothed, mask

def deform_state_filter(label, width):
    last_state_idx = 0
    state = 3
    state_tmp = 3
    count = 0
    label_refine = np.ones_like(label)*3
    for idx in range(len(label)):
        if state_tmp == label[idx]:
            count = count + 1
            if idx == len(label) - 1:
                if count >= width:
                    label_refine[last_state_idx:] = state_tmp
                else: 
                    label_refine[last_state_idx:] = state
        else:
            if count >= width:
                label_refine[last_state_idx:idx] = state_tmp
                last_state_idx = idx
                state = state_tmp
            if idx == len(label) - 1: 
                label_refine[last_state_idx:] = state
            state_tmp = label[idx]
            count = 1
    return label_refine

def cpd_locate(label):
    state_tmp = 3
    last_cpd = 0
    cpd_array = np.zeros_like(label)
    for idx in range(len(label)):
        if idx == 0:
            state_tmp = label[idx]
        elif state_tmp != label[idx]:
            cpd_array[idx] = 1
            state_tmp = label[idx]
            last_cpd = idx
    return cpd_array, last_cpd

def daterange(date1, date2, delta_days):
    for day_num in range(0, int((date2 - date1).days) + 1, delta_days):
        yield date1 + timedelta(day_num)


if __name__ == '__main__':

    params = {'batch_size': 6400, # 32 * 100 batch size for training
            'dim': 63, # time step
            'n_classes': 5,
            'n_samples': 1,
            'n_channels': 1,
            'shuffle': False}

    # text or csv path
    # data_path = "./data/deformation.txt"
    data_path = "./meida.csv"

    data_file_name = os.path.basename(data_path).split('.')[0]
    data_dir_name = os.path.dirname(data_path)
    plot_path = os.path.join(data_dir_name, 'plot')
    if not os.path.isdir(plot_path):
        os.makedirs(plot_path)

    # ts_df = pd.read_csv(data_path, delim_whitespace=True) # for txt
    ts_df = pd.read_csv(data_path) # for csv

    ts_df_cp = ts_df.copy()
    date_str_list = list(ts_df)
    print(ts_df.head())  # (166765, 121)

    first_date_str = date_str_list[0]
    last_date_str = date_str_list[-1]

    first_date = parse(first_date_str)
    last_date = parse(last_date_str)
    delta_days = timedelta(days=12)

    print(first_date+delta_days, last_date)

    period = (last_date - first_date).days
    date_num = int(period / delta_days.days + 1)
    print('period days and gap numbers:', period, date_num)

    first_date_value = ts_df_cp[first_date_str].values
    mask = np.ones(date_num)

    for i_dt, dt in enumerate(daterange(first_date, last_date, delta_days.days)):
        dt_str = dt.strftime("%Y%m%d")
        if (dt_str in date_str_list):
            # print("got", dt_str)
            ts_df[dt_str] = ts_df[dt_str] - first_date_value
        else:
            # print('miss', dt_str)
            ts_df[dt_str] = 0
            mask[i_dt] = 0

    ts_df.sort_index(axis=1, inplace=True)  # sort the data based on the column value
    date_str_list_full = list(ts_df)
    print(ts_df)
    print(mask)    
    mask_idx = mask > 0
    date_list_full = [parse(date_str).date() for date_str in date_str_list_full]
    print(date_str_list_full, 'len', len(date_str_list_full))
    date_list= [parse(date_str).date() for date_str in date_str_list]

    X_noise = ts_df.loc[:, first_date_str:last_date_str].values  # type numpy.ndarray

    batch_size = params['batch_size']
    dim = date_num
    n_channels = params['n_channels']

    len_df = len(ts_df.index)
    X_mask = np.empty((len_df, dim, n_channels))
    X_mask[:, :, 0] = mask # may broadcasting

    X_timestamp = np.empty((len_df, dim, 1))
    X_timestamp[:, :, 0] = np.arange(dim)  # np.arange(self.dim) shape (self.dim,) may broadcasting


    model_ch1 = load_mask_model("./model_select/model_mask_run_63.h5") # dim 63 model

    X_noise_scaled = scale_normalize(X_noise)

    X_input = np.empty((len_df, dim, n_channels))

    X_input[:, :, 0] = X_noise_scaled
    y_trend, y_seasonal, _, y_label_oh = model_ch1.predict([X_input, X_mask, X_timestamp], batch_size=batch_size)


    print(y_trend.shape)  # (4487, 135, 1)
    y_trend = np.squeeze(y_trend, axis=-1)  # (4487, 135)
    y_seasonal = np.squeeze(y_seasonal, axis=-1)
    y_label = np.argmax(y_label_oh, axis=-1)

    df_trend = ts_df.copy()
    df_trend.loc[:, first_date_str:last_date_str] = y_trend
    df_trend['s_var'] = np.var(y_seasonal, axis=1)
    df_trend['s_mean'] = np.mean(y_seasonal, axis=1)
    
    # global acc_index
    pred_v = y_trend[:, :-3] - y_trend[:,3:]
    pred_delta_v = (pred_v[:, :-3] - pred_v[:,3:])*np.sign(pred_v[:, :-3])
    df_trend['dv_sum'] = np.mean(pred_delta_v, axis=-1)
    df_trend['v_sum'] = np.mean(pred_v, axis=-1)
    y_dv_sum_last = np.mean(pred_delta_v, axis=-1)
    y_v_sum_last = np.mean(pred_v, axis=-1)
    print(df_trend.head(5))
    # label refine
    idx_deform_large = (y_trend[:, 0] - y_trend[:, -1]) > 10 # deform > 10mm
    idx_got_cpd = (np.max(y_label, axis=-1) - np.min(y_label, axis=-1)) > 0 # have change point
    y_label_refine = y_label.copy()
    y_cpd = np.zeros_like(y_label)
    idx_select = idx_deform_large & idx_got_cpd
    y_trend_select = y_trend[idx_select, :]
    y_label_select = y_label[idx_select, :]
    y_cpd_select = np.zeros_like(y_label_select)
    # last_cpd = np.zeros(y_label_select[0])
    y_dv_sum_last_select = np.zeros(y_label_select.shape[0])
    y_v_sum_last_select = np.zeros(y_label_select.shape[0])
    for i_select in range(y_label_select.shape[0]):
        y_label_select[i_select, :] = deform_state_filter(y_label_select[i_select, :], 15)
        y_cpd_select[i_select, :], last_cpd = cpd_locate(y_label_select[i_select, :])
        pred_v = y_trend_select[i_select, last_cpd:-3] - y_trend_select[i_select, last_cpd+3:]
        pred_delta_v = (pred_v[:-3] - pred_v[3:])*np.sign(pred_v[:-3])
        y_dv_sum_last_select[i_select] = np.mean(pred_delta_v)
        y_v_sum_last_select[i_select] = np.mean(pred_v)
    
    y_label_refine[idx_select, :] = y_label_select
    y_cpd[idx_select, :] = y_cpd_select
    y_dv_sum_last[idx_select, ] = y_dv_sum_last_select
    y_v_sum_last[idx_select, ] = y_v_sum_last_select

    df_trend['dv_sum_l'] = y_dv_sum_last
    df_trend['v_sum_l'] = y_v_sum_last

    df_label = df_trend.copy()
    df_label.loc[:, first_date_str:last_date_str] = y_label
    df_label_refine = df_trend.copy()
    df_label_refine.loc[:, first_date_str:last_date_str] = y_label_refine
    df_cpd = df_trend.copy()
    df_cpd.loc[:, first_date_str:last_date_str] = y_cpd
    df_seasonal = df_trend.copy()
    df_seasonal.loc[:, first_date_str:last_date_str] = y_seasonal

    data_trend = df_trend.loc[:, first_date_str:last_date_str].values  # type numpy.ndarray
    data_seasonal = df_seasonal.loc[:, first_date_str:last_date_str].values  # type numpy.ndarray    
    data_label_refine = df_label_refine.loc[0, first_date_str:last_date_str].values
    # print(y_label)
    # print(data_label_refine)
    cpd_vector = df_cpd.loc[0, first_date_str:last_date_str].values
    # print(cpd_vector)
    cpd_index = np.argwhere(cpd_vector)
    cpd_index = np.squeeze(cpd_index, -1)

    plt_linewidth = 2
    plt_markersize = 5
    fig = plt.figure(figsize=(8, 4))
    plt.plot(date_list, X_noise[0, mask_idx], 'k.--', linewidth=plt_linewidth, markersize=plt_markersize, label="Input TS")
    plt.plot(date_list_full, data_seasonal[0, :] + np.mean(data_trend)- (data_seasonal[0, :] + np.mean(data_trend))[0], 'b.-', linewidth=plt_linewidth, markersize=plt_markersize, label="Seasonal Est.")
    plt.plot(date_list_full, data_trend[0, :] - data_trend[0, :][0], 'm.-', linewidth=plt_linewidth, markersize=plt_markersize, label="Trend Est.")
    
    style_list = ['y*', 'go', 'r^']
    label_list = ['Linear', 'Decelerating', 'Accelerating']
    cpd_index_ext = np.concatenate((np.array([0]), cpd_index, np.array([len(cpd_vector)])), axis=0)
    # print(cpd_index_ext)
    for start, stop in zip(cpd_index_ext[:-1], cpd_index_ext[1:]):
        # print(start,stop)
        plt.plot(date_list_full[start:stop], 3*(data_label_refine[start:stop]+1) + np.max(data_trend), style_list[int(data_label_refine[start])], linewidth=plt_linewidth, markersize=plt_markersize, label=label_list[int(data_label_refine[start])])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="lower left", fontsize=10, fancybox=True, framealpha=0.5, prop=dict(weight='bold'))
    # plt.xlabel("Time", fontsize=8)
    plt.ylabel("Deformation (mm)", fontsize=10, fontweight='bold')    
    # plt.title("AccIndex: {:.2f}, Velocity: {:.2f}".format(dv_sum, dv), fontsize=12) # convert onehot to label num
    # plt.legend(loc="lower left", fontsize=10)
    plt.grid(True)
    fig.autofmt_xdate()    

    fig_name = os.path.join(plot_path, data_file_name)
    print(fig_name)
    save_fig(fig_name, fig_extension="png", transparent=True)
    plt.show()
    
    trend_shapefile_name = data_file_name + '_trend.csv'
    seasonal_shapefile_name = data_file_name + '_seasonal.csv'
    label_shapefile_name = data_file_name + '_label.csv'

    print("Saving trend shapefile: ", trend_shapefile_name)
    df_trend.to_csv(os.path.join(plot_path, trend_shapefile_name))
    print("Shapefile trend saved")

    print("Saving seasonal shapefile: ", seasonal_shapefile_name)
    df_seasonal.to_csv(os.path.join(plot_path, seasonal_shapefile_name))
    print("Shapefile seasonal saved")

    print("Saving label shapefile: ", label_shapefile_name)
    df_label_refine.to_csv(os.path.join(plot_path, label_shapefile_name))
    print("Shapefile label saved")