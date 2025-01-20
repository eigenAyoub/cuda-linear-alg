import io
import pandas as pd
import numpy as np
from PIL import Image

def extract_and_flatten(image_dict):
    """
    Given an image dictionary from the Parquet file (with keys 'bytes' and 'path'),
    load the image, convert it to grayscale, and return a flattened NumPy array of pixels.
    """
    img_bytes = image_dict['bytes']
    with io.BytesIO(img_bytes) as img_buffer:
        img = Image.open(img_buffer)
        # Convert to grayscale (if not already)
        img = img.convert("L")
    # Convert image to NumPy array and flatten it (e.g., 28x28 becomes 784-dimensional vector)
    return np.array(img).flatten()

# Load the MNIST Parquet files
train_df = pd.read_parquet('train.parquet')
test_df = pd.read_parquet('test.parquet')

# Apply the helper function to extract flattened pixel arrays
train_df['flat_pixels'] = train_df['image'].apply(extract_and_flatten)
test_df['flat_pixels'] = test_df['image'].apply(extract_and_flatten)

# Separate images and labels
# Assuming there is a column "label" in the DataFrame holding the class labels.
train_images = np.stack(train_df['flat_pixels'].to_numpy())  # shape: (N_train, 784)
train_labels = train_df['label'].to_numpy()

test_images = np.stack(test_df['flat_pixels'].to_numpy())  # shape: (N_test, 784)
test_labels = test_df['label'].to_numpy()

print("Train images shape:", train_images.shape)  # Expected: (N_train, 784)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)    # Expected: (N_test, 784)
print("Test labels shape:", test_labels.shape)

np.save('train_images.npy', train_images)
np.save('train_labels.npy', train_labels)
np.save('test_images.npy', test_images)
np.save('test_labels.npy', test_labels)
