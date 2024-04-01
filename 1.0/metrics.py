

import json
import os
import sys
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras import layers
import keras_nlp
import pandas as pd




def load_jsonl_to_memory(filepath, fraction=100):
    # Determine the total number of lines to calculate the size of the fraction
    with open(filepath, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)
    
    # Calculate the number of lines to process based on the fraction
    lines_to_process = total_lines // fraction

    #lines_to_process-= 3000000
    
    # Preallocate the list with None values for the fraction of data
    data = [None] * lines_to_process
    
    with open(filepath, 'r', encoding='utf-8') as file:
        processed_lines = 0  # Keep track of how many lines have been processed
        for index, line in enumerate(tqdm(file, total=total_lines, desc="Processing")):
            if (index+3) % fraction == 0:  # Process only every fraction-th line
                # Parse the JSON content from the line and add it to the data list
                data[processed_lines] = json.loads(line)
                processed_lines += 1
                if processed_lines >= lines_to_process:
                    break  # Stop if we've processed the intended number of lines
    
    return data

data = load_jsonl_to_memory('/mnt/datassd/processed_file.jsonl')

#|%%--%%| <sbzELpjvcg|oWVm17uamQ>

titles = [item['title'] for item in data]
view_counts = np.array([item['view_count'] for item in data])

# Parameters for tokenization and padding
vocab_size = 10000  # Adjust based on your dataset
max_length = 100  # Adjust based on the length of your titles
padding_type = 'post'
trunc_type = 'post'

# Initialize and fit the tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(titles)

# Convert titles to sequences and pad them
sequences = tokenizer.texts_to_sequences(titles)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Split the data into training and testing sets
X_train, X_test, y_train_n, y_test_n = train_test_split(padded_sequences, view_counts, test_size=0.2, random_state=42)


# Convert to log scale for normal distribution of data
y_test = np.log(y_test_n)
y_test = np.where(y_test == -np.inf, 0, y_test)

y_train = np.log(y_train_n)
y_train = np.where(y_train == -np.inf, 0, y_train)


# |%%--%%| <oWVm17uamQ|UZ5jTpSq96>

os.environ["KERAS_BACKEND"] = "tensorflow"

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

# Since we're working on a regression task, a GlobalMaxPooling1D layer is used to reduce the sequence dimension
x = layers.GlobalMaxPooling1D()(x)

# Additional dense layers for further processing
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(1, activation='linear')(x)  # Linear activation for a regression task

# Compile the model
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')

model.summary()

del data
del X_train, y_train

actual_model_name = 'YT_T90_log'

model = tf.keras.models.load_model(f'{actual_model_name}.keras')



#|%%--%%| <UZ5jTpSq96|ihiod14F64>

#from sklearn.model_selection import train_test_split
#
## Assuming X_test and y_test are your input arrays
#
## Round y_test values to the nearest integer if they are not already integers
#y_test_rounded = np.rint(y_test).astype(int)
#
## Determine the unique classes and their counts
#unique, counts = np.unique(y_test_rounded, return_counts=True)
#
## Find the minimum class size
#min_class_size = counts.min()
#
## Initialize lists to hold the sampled indices
#sampled_indices = []
#
## For each class, randomly sample min_class_size instances
#for class_value in unique:
#    class_indices = np.where(y_test_rounded == class_value)[0]
#    class_sample_indices = np.random.choice(class_indices, min_class_size, replace=False)
#    sampled_indices.extend(class_sample_indices)
#
## Convert to an array for fancy indexing
#sampled_indices = np.array(sampled_indices)
#
## Sample X_test and y_test using the indices
#X_test_sampled = X_test[sampled_indices]
#y_test_sampled = y_test[sampled_indices]
#
## X_test_sampled and y_test_sampled now contain uniformly sampled data
#

# |%%--%%| <ihiod14F64|A5CwqIiqkS>


import matplotlib.pyplot as plt

plt.hist(y_test_sampled)
plt.savefig('uniform.png')



#|%%--%%| <A5CwqIiqkS|y5pjSZNsDY>

plt.hist(y_test)
plt.savefig('y_test_dis.png')

# |%%--%%| <y5pjSZNsDY|NjVou6EDo0>



predictions = model.predict(X_test)

for i in range(10):  # Display first 10 predictions
    print(f"Predicted view count: {predictions[i]}, Actual view count: {y_test[i]}")


# #|%%--%%| <NjVou6EDo0|NWJd7aQ28G>

import seaborn as sns
import matplotlib.pyplot as plt


heatmap, xedges, yedges = np.histogram2d(y_test.flatten(), predictions.flatten(), bins=50)

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]


plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.savefig(f'{actual_model_name}_heatmap_uniform.png')




import matplotlib.pyplot as plt


plt.scatter(y_test, predictions, alpha=0.4)
plt.xlabel('Actual View Count')
plt.ylabel('Predicted View Count')
plt.savefig(f'{actual_model_name}_scatter_uniform.png')



# |%%--%%| <NWJd7aQ28G|9IRxPNoVbP>

