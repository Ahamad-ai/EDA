import pandas as pd
from sklearn.utils import resample

class DataPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load data from a CSV file."""
        self.data = pd.read_csv(self.file_path)
        return self.data

    def Upsample_Minority_Class_dataset(self):
        """Upsample Minority Class"""
        if self.data is None:
            raise ValueError("Data not loaded. Please load the data first.")

        # Identify the minority class
        class_counts = self.data['class'].value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        # Separate minority and majority classes
        minority_data = self.data[self.data['class'] == minority_class]
        majority_data = self.data[self.data['class'] == majority_class]

        # Upsample minority class
        upsampled_minority_data = minority_data.resample(minority_class,
                                                         replace=True,
                                                         n_samples=len(majority_data),
                                                            random_state=42)

        # Combine majority and upsampled minority data
        balanced_data = pd.concat([majority_data, upsampled_minority_data])

        return balanced_data
    
    def Downsample_Majority_Class_dataset(self):
        """Downsample Majority Class"""
        if self.data is None:
            raise ValueError("Data not loaded. Please load the data first.")

        # Identify the minority class
        class_counts = self.data['class'].value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()

        # Separate minority and majority classes
        minority_data = self.data[self.data['class'] == minority_class]
        majority_data = self.data[self.data['class'] == majority_class]

        # Downsample majority class
        downsampled_majority_data = majority_data.sample(n=len(minority_data), random_state=42)

        # Combine downsampled majority and minority data
        balanced_data = pd.concat([downsampled_majority_data, minority_data])

        return balanced_data