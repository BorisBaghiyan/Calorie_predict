# 🏃‍♂️ Predict Calorie Expenditure from Activity Data 🏃‍♀️

This project aims to predict the number of calories burned based on physical activity data using machine learning models. The goal is to develop a model that can accurately estimate calorie expenditure given various activity parameters.

## 🚀 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/BorisBaghiyan/Calorie_predict.git
    ```

2. Navigate to the project directory:
    ```bash
    cd Calorie_predict
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Launch Jupyter Notebook to explore and run the code:
    ```bash
    jupyter notebook
    ```

## 🔧 Technologies

This project uses the following technologies:

- **Python** 🐍: Programming language.
- **Pandas** 📊: Data manipulation and analysis.
- **NumPy** 🔢: Numerical computations.
- **Scikit-learn** 🔬: Machine learning library for model building.
- **Matplotlib & Seaborn** 📈: Data visualization.
- **Jupyter Notebook** 📓: Interactive coding environment.

## 📝 How to Use

1. **Prepare the dataset**:
    - Download the dataset from Kaggle: [Calorie Expenditure Dataset](https://www.kaggle.com/datasets/)
    - Place it inside the `data/` folder.

2. **Train the model**:
    - Launch Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the `calorie_predict.ipynb` notebook and run the cells sequentially for data loading, preprocessing, and model training.

3. **Make predictions**:
    - Use the trained model to predict calorie expenditure:
      ```bash
      python src/inference.py --input_data path/to/your/activity_data.csv
      ```

## 💡 Features

- 🔥 Predict the number of calories burned based on features like duration, heart rate, age, etc.
- 🔄 Data preprocessing: cleaning, scaling, and preparing the data.
- 📊 Model evaluation using metrics like Mean Squared Error (MSE) and R-squared (R²).
- 🌈 Visualization: training curves, prediction plots, feature importance.

## 🧠 Model Architecture

- **Input layer**: Activity features (duration, heart rate, age, weight, etc.)
- **Regression model**: Random Forest, Linear Regression, or other regressors.
- **Output layer**: Predicted number of calories burned.

## 🏆 Model Performance

- **Loss function**: Mean Squared Error (MSE)
- **Metrics**: R-squared (R²), Mean Squared Error (MSE)

## 📊 Visualizations

- Training loss curves
- Comparison of actual vs. predicted calorie values
- Feature importance charts

## 🤝 Contributing

Contributions are welcome:
- Fork the repository
- Open issues
- Submit pull requests

---
