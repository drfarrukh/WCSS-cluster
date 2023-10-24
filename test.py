# %%
import pandas as pd
import numpy as np

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

# Define the labels
labels = [
    "Benign", "DDOS attack-HOIC", "DoS attacks-Hulk", "Bot", "Infiltration",
    "SSH-Bruteforce", "DoS attacks-GoldenEye", "FTP-BruteForce",
    "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris", "DDOS attack-LOIC-UDP",
    "Brute Force -Web", "Brute Force -XSS", "SQL Injection"
]

# Number of instances
num_instances = 7841683

# Number of features
num_features = 76

# Create random data
data = np.random.rand(num_instances, num_features)

# %%

# Create a DataFrame
df = pd.DataFrame(data, columns=[f"Feature_{i+1}" for i in range(num_features)])

# Add the "Label" column with random labels
df['Label'] = np.random.choice(labels, num_instances)

print("df done")
# %%

df.shape
# %%
import gc
gc.collect()
# %%
# %%
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'].values.ravel())
df['Label'].unique()

unique_labels = df['Label'].unique()
original_labels = label_encoder.inverse_transform(unique_labels)

label_mapping = dict(zip(unique_labels, original_labels))
print(label_mapping)

# %%

Labels_in_df = df['Label'].unique()


# %%
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Split your data into features (X) and labels (y)
X = df.drop('Label', axis=1)  # Assuming you have features in your DataFrame
y = df['Label']


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
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

def train_and_evaluate(model, X_train, y_train, X_test, y_test, output_folder, model_name):
    # Train the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=128, validation_split=0.2, verbose=2)
    
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

CNN_LSTM2D = tf.keras.Sequential([
    tf.keras.layers.ConvLSTM2D(64, (3, 3), activation='relu', input_shape=(76, 1, 1), return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ConvLSTM2D(128, (3, 3), activation='relu', return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ConvLSTM2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
CNN_LSTM2D.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

CNN_LSTM2D.summary()
#%%

train_and_evaluate(CNN_LSTM2D, X_train, y_train, X_test, y_test, 'Output_plots', 'CNN-LSTM-2D')

#%%