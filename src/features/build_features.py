# Preprocess data
class Preprocessor():
    # Scale images
    def scaler(self, training_data, testing_data):
        self.training_data_scaled = training_data.map(lambda x, y: (x/255, y))
        self.testing_data_scaled = testing_data.map(lambda x, y: (x/255, y))
        
        return self.training_data_scaled, self.testing_data_scaled