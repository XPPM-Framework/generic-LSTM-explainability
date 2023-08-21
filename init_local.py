import warnings

import tensorflow as tf
from keras import backend as K
import argparse
from LSTM_sequence import prepare_data_predict_and_obtain_explanations
import pandas as pd
import os

import sys
import keras
import google.protobuf
import h5py
import shap

print("Python: " + str(sys.version_info[0]) + "." + str(sys.version_info[1]))
print("tensorflow: " + tf.__version__)
print("keras: " + keras.__version__)
print("protobuf: " + google.protobuf.__version__)
print("h5py: " + h5py.__version__)
print("shap: " + shap.__version__)


def create_experiment_name(filename, pred_column):
    filename = filename.replace('.csv', '').replace('data/', '')
    experiment_name = filename + "_" + pred_column
    return experiment_name

def read_data(filename, *, limit: float=None):
    if '.csv' in filename:
        try:
            df = pd.read_csv(filename, header=0)
        except UnicodeDecodeError:
            df = pd.read_csv(filename, header=0, encoding="cp1252")
    elif '.parquet' in filename:
        df = pd.read_parquet(filename, engine='pyarrow')

    # Limit df to 10% of its original size from the beginning
    if limit is not None:
        warnings.warn(f"Limiting dataframe to {int(limit * 100)}% of its original size")
        df = df.iloc[:int(len(df) * limit)]
    print("Read dataframe shape:", df.shape)
    return df

def create_folders_for_experiment(experiment_name):
    os.makedirs(experiment_name, exist_ok=True)
    os.makedirs(experiment_name + "/model", exist_ok=True)
    os.makedirs(experiment_name + "/shap", exist_ok=True)
    os.makedirs(experiment_name + "/results", exist_ok=True)
    os.makedirs(experiment_name + "/plots", exist_ok=True)


parser = argparse.ArgumentParser(description='6 mandatory parameters, 4 optional')
parser.add_argument('--mandatory', nargs=7, help='filename case_id_position start_date_position date_format pred_column experiment_name mode', required=True)
parser.add_argument('--shap', default=True)
parser.add_argument('--end_date_position', default=None)
parser.add_argument('--pred_attributes', default=None)
#if True retrains model and overrides previous one (if developer wants to train with different parameters)
parser.add_argument('--num_epochs', default=500, type=int)

args = parser.parse_args()

#mandatory parameters
mandatory = args.mandatory
filename = mandatory[0]
case_id_position = int(mandatory[1])
start_date_position = int(mandatory[2])
date_format = mandatory[3]
pred_column = mandatory[4]  # remaining_time if you want to predict that
experiment_name = mandatory[5]
mode = mandatory[6]
#experiment_name = create_experiment_name(filename, pred_column)

#optional parameters
shap_calculation = args.shap
pred_attributes = args.pred_attributes
end_date_position = args.end_date_position
override = False
num_epochs = args.num_epochs

if pred_attributes is not None:
    pred_attributes = pred_attributes.split(',')
if end_date_position is not None:
    end_date_position = int(end_date_position)


n_neurons = 100
n_layers = 8


# limit the quantity of memory you wanna use in your gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

# Size limitation of the dataset for testing (Not smart)
limit = 0.3

df = read_data(filename, limit=limit)
experiment_name = 'experiment_files/' + experiment_name
if mode == "train":
    # if override False create folders only if it is the first train
    if override is True or (override is False and not os.path.exists(experiment_name)):
        create_folders_for_experiment(experiment_name)

prepare_data_predict_and_obtain_explanations(experiment_name, df, case_id_position, start_date_position, date_format,
                                    end_date_position, pred_column, mode, pred_attributes, n_neurons, n_layers,
                                    shap_calculation, override, num_epochs)