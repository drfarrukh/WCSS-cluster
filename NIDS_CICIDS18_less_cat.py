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
from tensorflow.keras.callbacks import EarlyStopping

#graph
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



import gc

    
# %%

gpus = tf.config.list_physical_devices('GPU')
print(len(gpus))

for gpu in gpus:
    device = tf.config.PhysicalDevice(name=gpu.name, device_type='GPU')
    details = tf.config.experimental.get_device_details(device)
    print(details['device_name'])

# %%
df = pd.read_csv('preprocessed_data.csv', parse_dates=True, keep_date_col=True)
print(f'Instances: {df.shape[0]}')
print(f'Features: {df.shape[1]}')

# %%
df.describe()

# %%
df.head()

# %%
Labels_in_df = df['Label'].unique()
df['Label'].value_counts()

class_labels = df['Label'].value_counts().index.tolist()

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
plt.savefig('Output_plots/before_Downsampling_of_Benign_class_pie.png')
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
plt.savefig('Output_plots/before_Downsampling_of_Benign_class_bar.png')
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
plt.savefig('Output_plots/after_Downsampling_of_Benign_class_pie.png')
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
plt.savefig('Output_plots/after_Downsampling_of_Benign_class_bar.png')
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
# Apply SMOTE to balance the training set
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Check the new distribution of labels
print(y_train.value_counts())

gc.collect()

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
num_classes = len(Labels_in_df)
print(num_classes)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

def train_and_evaluate(model, X_train, y_train, X_test, y_test, output_folder, model_name):
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    # Train the model with early stopping
    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=2)
    
    # Create a DataFrame from the training history
    history_df = pd.DataFrame(history.history)
    # Save the DataFrame to a CSV file
    history_df.to_csv(f'{output_folder}/{model_name}_training_history.csv', index=False)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    # Plot training & validation accuracy and loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} - Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the plots to the output folder
    plt.savefig(f'{output_folder}/{model_name}_training_plots.png')
    plt.show()

    # Generate a classification report
    y_pred = model.predict(X_test, verbose=2)
    y_pred_classes = np.argmax(y_pred, axis=1)
    class_report = classification_report(y_test, y_pred_classes)

    # Generate a classification report
    y_pred = model.predict(X_test, verbose=2)
    y_pred_classes = np.argmax(y_pred, axis=1)
    class_report = classification_report(y_test, y_pred_classes)

    confusion = confusion_matrix(y_test, y_pred_classes)
    

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=Labels_in_df, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save the figure as a PNG file
    plt.savefig(f'{output_folder}/{model_name}_confusion_matrix.png')
    # Show the plot
    plt.show()

    # Print and save the classification report
    print(class_report)
    with open(f'{output_folder}/{model_name}_classification_report.txt', 'w') as report_file:
        report_file.write(class_report)

#%%


# %%
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# %%
# Build your CNN model using TensorFlow
Simple_1DCNN = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=(X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
Simple_1DCNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

Simple_1DCNN.summary()

train_and_evaluate(Simple_1DCNN, X_train, y_train, X_test, y_test, 'Output_plots', '1D-CNN')

#%%

# Reshape the data to 2D format
X_train = X_train.reshape(-1, 76, 1, 1)
X_test = X_test.reshape(-1, 76, 1, 1)

# Define the CNN model
Simple_2DCNN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(76, 1, 1)),
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(14, activation='softmax')  # 14 output classes for the labels
])

# Compile the model
Simple_2DCNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


train_and_evaluate(Simple_2DCNN, X_train, y_train, X_test, y_test, 'Output_plots', '2D-CNN')

#%%

# CNN_LSTM2D = tf.keras.Sequential([
#     tf.keras.layers.ConvLSTM2D(64, (3, 3), activation='relu', input_shape=(76, 1, 1), return_sequences=True),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.ConvLSTM2D(128, (3, 3), activation='relu', return_sequences=True),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.ConvLSTM2D(256, (3, 3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# # Compile the model
# CNN_LSTM2D.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# CNN_LSTM2D.summary()
# #%%

# train_and_evaluate(CNN_LSTM2D, X_train, y_train, X_test, y_test, 'Output_plots', 'CNN-LSTM-2D')

#%%