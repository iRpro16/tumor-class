from src.data.make_dataset import Dataset
from src.features.build_features import Preprocessor

if __name__ == "__main__":
    
    # Get Data
    data = Dataset()
    
    training, testing = data.get_datasets()
    
    preprocessor = Preprocessor()
    training_scaled, testing_scaled = preprocessor.scaler(training, testing)