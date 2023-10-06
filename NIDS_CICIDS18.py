# %%

# %%
import os
import json
import numpy as np
import pandas as pd
from pprint import pprint
from itertools import combinations, product

#sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.metrics import precision_recall_curve, auc, roc_curve, recall_score, precision_score, f1_score

#tensorflow
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#graph
import seaborn as sns
import matplotlib.pyplot as plt


import gc

    
# %%

gpus = tf.config.list_physical_devices('GPU')
print(len(gpus))

for gpu in gpus:
    device = tf.config.PhysicalDevice(name=gpu.name, device_type='GPU')
    details = tf.config.experimental.get_device_details(device)
    print(details['device_name'])

# %%
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)
                # else:
                #     df[col] = 0.0  # Set values larger than float64 to zero
        else:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    gc.collect()
    return df

# %%

import os

folder_path = './kaggle_data'  # Replace with the actual folder path

# Get a list of all files in the folder
files_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# files_list =['Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
#              'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
             
#              ]

# %%
dataset_csv_path = './kaggle_data'

complete_paths = []
for csv_file_name in files_list:
    complete_paths.append(os.path.join(dataset_csv_path, csv_file_name))

df = pd.concat(map(import_data, complete_paths), 
               ignore_index = True)

print("Dataset Loaded")

# %%
df['Label'].value_counts()

# %%
def clean_df(df):
    # Remove the space before each feature names
    df.columns = df.columns.str.strip()
    print('dataset shape', df.shape)

    # This set of feature should have >= 0 values
    num = df._get_numeric_data()
    num[num < 0] = 0

    zero_variance_cols = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            zero_variance_cols.append(col)
    df.drop(zero_variance_cols, axis = 1, inplace = True)
    print('zero variance columns', zero_variance_cols, 'dropped')
    print('shape after removing zero variance columns:', df.shape)

    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    print(df.isna().any(axis = 1).sum(), 'rows dropped')
    df.dropna(inplace = True)
    print('shape after removing nan:', df.shape)

    # Drop duplicate rows
    df.drop_duplicates(inplace = True)
    print('shape after dropping duplicates:', df.shape)

    column_pairs = [(i, j) for i, j in combinations(df, 2) if df[i].equals(df[j])]
    ide_cols = []
    for column_pair in column_pairs:
        ide_cols.append(column_pair[1])
    df.drop(ide_cols, axis = 1, inplace = True)
    print('columns which have identical values', column_pairs, 'dropped')
    print('shape after removing identical value columns:', df.shape)
    gc.collect()
    return df

# %%
df = clean_df(df)
print(f'Instances: {df.shape[0]}')
print(f'Features: {df.shape[1]}')

# %%
df.describe()

# %%
df.head()

# %%
Labels_in_df = df['Label'].unique()
df['Label'].value_counts()

# %%
rows_with_header = df[df['Label'].str.contains('Label')]
rows_with_header

# %%
df = df[~df['Label'].str.contains('Label')]
df.shape

# %%
dropping_cols = ['Dst Port', 'Protocol', 'Timestamp']
df.drop(dropping_cols, axis = 1, inplace = True)

Labels_in_df = df['Label'].unique()
num_classes = len(Labels_in_df)
df['Label'].value_counts()

# %%
import matplotlib.pyplot as plt

# Your code to create the pie chart
df['Label'].value_counts().plot(kind='pie', figsize=(6, 6), autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])

# Adding a title and labels
plt.title('Distribution of Labels before Downsampling of "Benign" Class')
plt.ylabel('')  # Removing the default ylabel

# Adding a legend
plt.legend(labels=df['Label'].value_counts().index, loc='upper right')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Display the plot
plt.show()


# %%
import matplotlib.pyplot as plt

# Your code to create a bar chart for label counts
label_counts = df['Label'].value_counts()
plt.figure(figsize=(10, 6))
plt.barh(label_counts.index, label_counts.values, color='skyblue')

# Adding labels and a title
plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Distribution of Labels before Downsampling of "Benign" Class')

# Display the plot
plt.tight_layout()
plt.show()


# %%
# Calculate the count of 'Benign' label
benign_count = len(df[df['Label'] == 'Benign'])

# Calculate the count of all other labels combined
other_labels_count = len(df) - benign_count

# Check if 'Benign' count is higher than all other labels combined
if benign_count > other_labels_count:
    # Find the next highest label by grouping and counting
    label_counts = df['Label'].value_counts()
    next_highest_label = label_counts.index[1]  # Index 1 corresponds to the next highest label
    
    # Downsample 'Benign' to match the count of the next highest label
    df = df.groupby('Label', group_keys=False).apply(lambda x: x.sample(n=min(len(x), len(df[df['Label'] == next_highest_label]))))
    
# Print the modified DataFrame
print(df['Label'].value_counts())


# Shuffle the dataframe to mix the data
df = df.sample(frac=1, random_state=42)


# Display the new distribution of labels
print(df['Label'].value_counts())

# %%
import matplotlib.pyplot as plt

# Your code to create the pie chart
df['Label'].value_counts().plot(kind='pie', figsize=(6, 6), autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])

# Adding a title and labels
plt.title('Distribution of Labels after Downsampling of "Benign" Class')
plt.ylabel('')  # Removing the default ylabel

# Adding a legend
plt.legend(labels=df['Label'].value_counts().index, loc='upper right')

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Display the plot
plt.show()


# %%
import matplotlib.pyplot as plt

# Your code to create a bar chart for label counts
label_counts = df['Label'].value_counts()
plt.figure(figsize=(10, 6))
plt.barh(label_counts.index, label_counts.values, color='skyblue')

# Adding labels and a title
plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Distribution of Labels after Downsampling of "Benign" Class')

# Display the plot
plt.tight_layout()
plt.show()


# %%
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'].values.ravel())
df['Label'].unique()

unique_labels = df['Label'].unique()
original_labels = label_encoder.inverse_transform(unique_labels)

label_mapping = dict(zip(unique_labels, original_labels))
print(label_mapping)


# %%
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split your data into features (X) and labels (y)
X = df.drop('Label', axis=1)  # Assuming you have features in your DataFrame
y = df['Label']


# %%
print(f'Intial values with Inf:          {X.shape}')

# Convert X to the appropriate data type
X = X.astype(np.float64)

max_float64 = np.finfo(np.float64).max

# Now perform the comparison
rows_to_drop = X[(X > max_float64).any(axis=1)].index

# max_float64 = np.finfo(np.float64).max  # Maximum finite value for float64

# # Find rows where any feature column contains a value larger than float64 maximum
# rows_to_drop = X[(X > max_float64).any(axis=1)].index

# Drop rows with values larger than float64 maximum from both X_initial and y_initial
X = X.drop(rows_to_drop)
y = y.drop(rows_to_drop)

print(f'Intial values after Inf removal: {X.shape}')

# %%
gc.collect()

# %%
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(f"Normalized shape: {X.shape} \n")

# %%
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')

print(f'Shape of  X_test: {X_test.shape}')
print(f'Shape of  y_test: {y_test.shape}')


# %%
# # # Apply SMOTE to balance the training set
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# # Check the new distribution of labels
# print(y_train.value_counts())

# gc.collect()

# %%
# # Your code to create a bar chart for label counts
# label_counts = df['Label'].value_counts()
# plt.figure(figsize=(10, 6))
# plt.barh(label_counts.index, label_counts.values, color='skyblue')

# # Adding labels and a title
# plt.xlabel('Count')
# plt.ylabel('Label')
# plt.title('Distribution of Labels after Downsampling of "Benign" Class')

# # Display the plot
# plt.tight_layout()
# plt.show()

# %%
print(f'Type of X_train: {type(X_train)}')

if not isinstance(y_train, np.ndarray):
    # If not, convert it to a NumPy array
    y_train = y_train.to_numpy()

print(f'Type of y_train: {type(y_train)}')


print(f'Type of X_test: {type(X_test)}')

if not isinstance(y_test, np.ndarray):
    # If not, convert it to a NumPy array
    y_test = y_test.to_numpy()

print(f'Type of y_train: {type(y_test)}')

# %%
num_classes = len(Labels_in_df)
print(num_classes)

# %%
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# %%
# Build your CNN model using TensorFlow
model1 = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model1.summary()

# %%
# Train the model
history1 = model1.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model1.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# %%


# %%
import tensorflow as tf

# Build your LSTM model using TensorFlow
model2 = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.LSTM(50, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(200, activation='relu', return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model2.summary()


# %%
# Train the model
history2 = model2.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model2.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# %%


# %%
import tensorflow as tf

# Build your CNN-LSTM model using TensorFlow
model3 = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(X_train.shape[1], 1, 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.ConvLSTM2D(50, kernel_size=(3, 1), activation='relu', return_sequences=True),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.ConvLSTM2D(100, kernel_size=(3, 1), activation='relu', return_sequences=True),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.ConvLSTM2D(200, kernel_size=(3, 1), activation='relu', return_sequences=True),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model3.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model3.summary()


# %%
# Train the model
history3 = model3.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# %%


# %%



