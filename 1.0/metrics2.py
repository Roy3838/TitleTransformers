
import os
os.environ["KERAS_BACKEND"] = "jax"
# |%%--%%| <Y1ps5BQSbz|oWVm17uamQ>

import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def data_generator(filepath, tokenizer, batch_size=32, max_length=100, test_size=0.2):
    """
    A generator function that yields batches of tokenized and padded sequences and their labels.
    
    Parameters:
    - filepath: Path to the JSONL file.
    - tokenizer: An instance of tf.keras.preprocessing.text.Tokenizer.
    - batch_size: The number of samples to return in each batch.
    - max_length: The maximum length of the sequences after padding.
    - test_size: The proportion of the dataset to include in the test split.
    
    Yields:
    - A tuple (batch_sequences, batch_labels), where:
        - batch_sequences is a numpy array of tokenized and padded sequences.
        - batch_labels is a numpy array of labels for each sequence in the batch.
    """
    titles = []
    view_counts = []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading and processing data"):
            record = json.loads(line)
            titles.append(record['title'])
            view_counts.append(record['view_count'])
            
            if len(titles) == batch_size:
                sequences = tokenizer.texts_to_sequences(titles)
                padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
                labels = np.log(np.array(view_counts, dtype=np.float32))
                labels = np.where(labels == -np.inf, 0, labels)
                
                yield padded_sequences, labels
                
                titles = []
                view_counts = []
                
    if titles:
        sequences = tokenizer.texts_to_sequences(titles)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        labels = np.log(np.array(view_counts, dtype=np.float32))
        labels = np.where(labels == -np.inf, 0, labels)
        
        yield padded_sequences, labels

def sample_titles(filepath, sample_size=1000):
    """
    Reads a sample of titles from a JSONL file.

    Parameters:
    - filepath: Path to the JSONL file.
    - sample_size: Number of titles to sample.
    
    Returns:
    - A list of sampled titles.
    """
    titles = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            if len(titles) >= sample_size:
                break
            record = json.loads(line)
            titles.append(record['title'])
    return titles


max_length = 100
filepath = '/mnt/datassd/processed_file.jsonl'

titles_sample = sample_titles(filepath)


tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")

tokenizer.fit_on_texts(titles_sample)

# |%%--%%| <oWVm17uamQ|UZ5jTpSq96>

import mlflow
import keras
import keras_nlp
from keras import layers

vocab_size = 10000  # Adjust based on your vocabulary size
embedding_dim = 256
max_length = 100  # Adjust based on your titles' maximum length
num_heads = 8  # Number of attention heads in the Transformer encoder
intermediate_dim = 512  # Dimensionality of the encoder's intermediate (feed-forward) layer

# Define input layer
inputs = keras.Input(shape=(max_length,), dtype='int64')

# Token and position embedding layer
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=vocab_size,
    sequence_length=max_length,
    embedding_dim=embedding_dim,
)
x = embedding_layer(inputs)

# Transformer encoder layer
encoder = keras_nlp.layers.TransformerEncoder(
    num_heads=num_heads,
    intermediate_dim=intermediate_dim,
    activation='relu',
    dropout=0.1,
)
x = encoder(x)

# GlobalMaxPooling1D layer for regression task
x = layers.GlobalMaxPooling1D()(x)

# Additional dense layers
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x)  # Linear activation for regression

# Compile the model
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')
model.summary()

# |%%--%%| <UZ5jTpSq96|A5CwqIiqkS>


# MLflow tracking
mlflow.autolog()

epochs = 10
batch_size = 1024  # This is now only for your reference and generator configuration

with mlflow.start_run():
    # Assuming your data_generator now correctly configures batches of size `batch_size`
    model.fit(x=data_generator(filepath, tokenizer, batch_size, max_length),
              epochs=epochs,
              steps_per_epoch=83419,verbose = 2)  # Make sure this matches your actual number of batches per epoch)

    # Log additional metrics or parameters
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("embedding_dim", embedding_dim)
    mlflow.log_param("max_length", max_length)
    mlflow.log_param("num_heads", num_heads)
    mlflow.log_param("intermediate_dim", intermediate_dim)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("generator_batch_size", batch_size)  # Renamed to clarify this is the generator's batch size
    
    # Save and log the model in MLflow
    model_name = "YT_Transformer"
    model_path = f"{model_name}.keras"
    model.save(model_path)
    mlflow.keras.log_model(model, "model", registered_model_name=model_name)



# |%%--%%| <A5CwqIiqkS|NjVou6EDo0>

#model.save(f'{actual_model_name}.keras')
model = keras.saving.load_model('YT_Transformer.keras')
#
predictions = model.predict(data_generator(filepath, tokenizer))
#
#for i in range(10):  # Display first 10 predictions
#    print(f"Predicted view count: {predictions[i]}, Actual view count: {y_test[i]}")
#
# |%%--%%| <NjVou6EDo0|nNomoUvqzW>

# Make a line 
x = np.linspace(0,10,100)
y = np.linspace(0,10,100)

# |%%--%%| <nNomoUvqzW|NWJd7aQ28G>

import seaborn as sns
import matplotlib.pyplot as plt


heatmap, xedges, yedges = np.histogram2d(y_test.flatten(), predictions.flatten(), bins=100)

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.plot(x,y, 'r--')
plt.xlabel('Views Order of Magnitude')
plt.ylabel('Predicted Order of Magnitude')
plt.xlim(2,15)
plt.ylim(0,17)
plt.savefig(f'{actual_model_name}_heatmap_bonito.png')

#import matplotlib.pyplot as plt

#
#plt.scatter(y_test, predictions, alpha=0.1, s=0.5)
#plt.plot(x,y,'r--')
#plt.xlabel('Actual View Count')
#plt.ylabel('Predicted View Count')
#plt.savefig(f'{actual_model_name}_scatter_bonit.png')

# |%%--%%| <NWJd7aQ28G|v61GjURvbq>

# If you need to convert an array of values
y_test_e = np.exp(y_test)  # Assuming y_test was in loge form
y_test_10 = np.log10(y_test_e)

predictions_e = np.exp(predictions)  # Assuming predictions were in loge form
predictions_10 = np.log10(predictions_e)

y_test = y_test_10
predictions = predictions_10

# |%%--%%| <v61GjURvbq|ohrretbeKu>

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Assuming y_test and predictions are available and in log form
# Heatmap
heatmap, xedges, yedges = np.histogram2d(y_test.flatten(), predictions.flatten(), bins=100)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.figure(figsize=(10, 8))
sns.set(style="white")

# Using a colormap (e.g., 'viridis' which is visually appealing and colorblind-friendly)
plt.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')

# Assuming x, y for the red dashed line are defined correctly and correspond to log scale
plt.plot(x, y, 'r--')

plt.xlabel('Log of Actual View Count')
plt.ylabel('Log of Predicted View Count')
plt.colorbar(label='Count of Test')
plt.title('Heatmap of Predictions vs Actual Views')
plt.xlim(0, 9)
plt.ylim(0, 9)

# Adjusting x and y axis to show in 10^ format
ax = plt.gca()
ax.set_xticklabels([f'$10^{{{int(float(label))}}}$' for label in ax.get_xticks()])
ax.set_yticklabels([f'$10^{{{int(float(label))}}}$' for label in ax.get_yticks()])

plt.savefig(f'{actual_model_name}_heatmap_bonito.png')

# |%%--%%| <ohrretbeKu|sXLwhEE9CR>

plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")

# Scatter plot with adjustments for alpha and size for better visibility
plt.scatter(y_test, predictions, alpha=0.2, s=10, cmap='viridis')

plt.plot(x, y, 'r--')  # Assuming x, y for the red dashed line are correct

plt.xlabel('Log of Actual View Count')
plt.ylabel('Log of Predicted View Count')
plt.title('Scatter Plot of Predicted vs Actual Views')
plt.xlim(0, 9)
plt.ylim(0, 9)


# Adjust axis to reflect 10^x and 10^y
ax = plt.gca()
ax.set_xticklabels([f'$10^{{{int(float(label))}}}$' for label in ax.get_xticks()])
ax.set_yticklabels([f'$10^{{{int(float(label))}}}$' for label in ax.get_yticks()])

plt.savefig(f'{actual_model_name}_scatter_bonito.png')

# |%%--%%| <sXLwhEE9CR|9IRxPNoVbP>


