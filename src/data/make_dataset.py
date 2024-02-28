import tensorflow as tf
import keras

class Dataset():
    # Get dataset
    def get_datasets(self):
        self.training_ds = keras.utils.image_dataset_from_directory('brain-mri/Training', batch_size=32, shuffle=True)
        self.testing_ds = keras.utils.image_dataset_from_directory('brain-mri/Testing', batch_size=32, shuffle=True)
        
        return self.training_ds, self.testing_ds