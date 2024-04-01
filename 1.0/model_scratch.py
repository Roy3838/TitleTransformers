
import json
import os
import sys
from tqdm import tqdm

def load_jsonl_to_memory(filepath, fraction=4):
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

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming 'data' is your list of dictionaries
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


import numpy as np
import keras
from keras import layers
import keras_nlp

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

import tensorflow as tf
import keras
#del X_train, y_train

prev_model_name = 'YT_T90_log'
actual_model_name = 'YT_T90_log'

model = tf.keras.models.load_model(f'{prev_model_name}.keras')



# |%%--%%| <UZ5jTpSq96|A5CwqIiqkS>

# Assuming X_train, y_train are your training data and labels, respectively
#model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=32)


# |%%--%%| <A5CwqIiqkS|NjVou6EDo0>

#model.save(f'{actual_model_name}.keras')

predictions = model.predict(X_test)

for i in range(10):  # Display first 10 predictions
    print(f"Predicted view count: {predictions[i]}, Actual view count: {y_test[i]}")


#|%%--%%| <NjVou6EDo0|nNomoUvqzW>

# Make a line 
x = np.linspace(0,10,100)
y = np.linspace(0,10,100)

#|%%--%%| <nNomoUvqzW|NWJd7aQ28G>

#import seaborn as sns
#import matplotlib.pyplot as plt
#
#
#heatmap, xedges, yedges = np.histogram2d(y_test.flatten(), predictions.flatten(), bins=100)
#
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
#
#plt.imshow(heatmap.T, extent=extent, origin='lower')
#plt.plot(x,y, 'r--')
#plt.xlabel('Views Order of Magnitude')
#plt.ylabel('Predicted Order of Magnitude')
#plt.xlim(2,15)
#plt.ylim(0,17)
#plt.savefig(f'{actual_model_name}_heatmap_bonito.png')
#
#import matplotlib.pyplot as plt
#
#
#plt.scatter(y_test, predictions, alpha=0.1, s=0.5)
#plt.plot(x,y,'r--')
#plt.xlabel('Actual View Count')
#plt.ylabel('Predicted View Count')
#plt.savefig(f'{actual_model_name}_scatter_bonit.png')



#|%%--%%| <NWJd7aQ28G|v61GjURvbq>

# If you need to convert an array of values
y_test_e = np.exp(y_test)  # Assuming y_test was in loge form
y_test_10 = np.log10(y_test_e)

predictions_e = np.exp(predictions)  # Assuming predictions were in loge form
predictions_10 = np.log10(predictions_e)

y_test = y_test_10
predictions = predictions_10

#|%%--%%| <v61GjURvbq|ohrretbeKu>

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


#|%%--%%| <ohrretbeKu|sXLwhEE9CR>

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

