from src.data.make_dataset import Dataset
from src.features.build_features import Preprocessor
from src.models.evaluate_model import Evaluate
import numpy as np

if __name__ == "__main__":
    
    # Get Data
    data = Dataset()
    
    # Get training and testing batches
    training, testing = data.get_datasets()
    
    # Preprocessor
    preprocessor = Preprocessor()
    
    # Scale the data for values between 0-1
    training_scaled, testing_scaled = preprocessor.scaler(training, testing)
    
    # Predict with trained model
    evaluator = Evaluate("/home/irpro16/ml-projects/tumor-class/models/my_model9_76_4")
    
    # Get loss
    loss, acc = evaluator.evaluate(testing_scaled)
    
    # Metrics
    print(f"Loss: {loss:.3f}")
    print(f"Accuracy: {acc*100:.3f}%")
    