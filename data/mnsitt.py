import pandas as pd
import numpy as np
from PIL import Image
import io

def save_images_and_labels_to_binary(df, output_path):

    img_file_path = f'{output_path}_images.bin'
    lbl_file_path = f'{output_path}_labels.bin'
    
    with open(img_file_path, 'wb') as img_file, open(lbl_file_path, 'wb') as lbl_file:
        for _, row in df.iterrows():
            try:
                # Decode PNG bytes to an image
                image_data = row['image']['bytes']
                image = Image.open(io.BytesIO(image_data))

                image_array = np.array(image, dtype=np.uint8).flatten()

                img_file.write(image_array.tobytes())
                lbl_file.write(np.array([row['label']], dtype=np.uint8))
            except Exception as e:
                print(f"Failed to process an image: {e}")
                continue  # Skip this image and continue with the next

train_df = pd.read_parquet('../mlp/train-00000-of-00001.parquet')
test_df = pd.read_parquet('../mlp/test-00000-of-00001.parquet')

save_images_and_labels_to_binary(train_df, 'train_mnist')
save_images_and_labels_to_binary(test_df, 'test_mnist')
