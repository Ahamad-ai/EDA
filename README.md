# Data Science EDA Projects

This repository contains two main exploratory data analysis (EDA) projects:

- **Titanic EDA**: Classic EDA on the Titanic dataset.
- **Handling Imbalanced Datasets**: Techniques for addressing class imbalance using the Credit Card Fraud Detection dataset, including upsampling, downsampling, and SMOTE.

---

## Project Structure

```
EDA/
├── Handling_Imbalance_Datasets/
│   ├── data/
│   │   ├── raw/
│   │   │   └── creditcard.csv
│   │   └── processed/
│   │       ├── creditcard_upsampled.csv
│   │       └── creditcard_downsampled.csv
│   └── notebooks/
│       ├── experiments.ipynb
│       └── SMOTE.ipynb
├── titanic-eda/
│   ├── data/
│   ├── notebooks/
│   └── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 1. Titanic EDA

Exploratory data analysis on the Titanic dataset, focusing on understanding survival patterns and feature relationships.

### Typical Steps:
- Data loading and inspection
- Missing value analysis
- Statistical summaries
- Visualization of key variables (e.g., survival rate, age, class)
- Initial insights and data quality checks

_Notebooks and data are located in `titanic-eda/`._

---

## 2. Handling Imbalanced Datasets: Credit Card Fraud Example

Demonstrates upsampling, downsampling, and SMOTE techniques to address class imbalance in the Credit Card Fraud Detection dataset.

### experiments.ipynb covers:
- Data loading and class distribution analysis
- Upsampling the minority class using `sklearn.utils.resample`
- Downsampling the majority class
- Visualizations comparing class distributions before and after resampling
- Exporting balanced datasets for further modeling

### SMOTE.ipynb covers:
- Applying SMOTE (Synthetic Minority Over-sampling Technique) using `imblearn.over_sampling.SMOTE`
- Creating a balanced dataset with synthetic samples
- Visualizing the effect of SMOTE on feature space
- Preparing the data for further modeling

_Notebooks and data are located in `Handling_Imbalance_Datasets/`._

---

## Requirements

Install dependencies using:

```sh
pip install -r requirements.txt
```

Main libraries used:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- jupyter

---

## How to Run

1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
4. Open and run notebooks in the respective project folders.

---

## License

This project is for educational purposes.