import cv2
import numpy as np

# Preprocess data
class Preprocessor():
    # Scale images
    def scaler(self, training_data, testing_data):
        self.training_data_scaled = training_data.map(lambda x, y: (x/255, y))
        self.testing_data_scaled = testing_data.map(lambda x, y: (x/255, y))
        
        return self.training_data_scaled, self.testing_data_scaled
    
    
    # Preprocess an image for prediction
    def preprocess_image(self, img):
        # Resize
        self.resized_img = cv2.resize(img, (256, 256))
        # Normalize
        self.normalized_img = cv2.normalize(self.resized_img, None, 0, 1.0, 
                                       cv2.NORM_MINMAX, 
                                       dtype=cv2.CV_32F)
        # Expand dimension
        self.preprocessed_img = np.expand_dims(self.normalized_img, axis=0)
        
        return self.preprocessed_img