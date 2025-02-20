# Health-Predictor

Health-Predictor is a machine learning project designed to predict a health score for individuals based on daily metrics such as steps taken, calories burned, and sleep hours. This project leverages a feedforward neural network implemented with PyTorch and includes an interactive Dash dashboard for real-time predictions.

---

## **Features**
- **Health Score Prediction**: Predicts an individual's health score using daily activity metrics.
- **PyTorch Neural Network**: Built using a feedforward neural network for regression tasks.
- **Interactive Dashboard**: A user-friendly Dash application for inputting metrics and receiving predictions.

---

## **Requirements**
To run this project, you'll need the following:
- Python (>= 3.8)
- Required libraries:
  - PyTorch
  - pandas
  - numpy
  - scikit-learn
  - Dash
  - matplotlib (optional for visualizations)

Install all dependencies using:
```bash
pip install torch pandas numpy scikit-learn dash matplotlib joblib
```

---

## **Dataset**
The dataset consists of daily activity metrics:
- **Inputs**:
  - `steps`: Number of steps taken in a day.
  - `calories_burned`: Calories burned during the day.
  - `sleep_hours`: Total hours of sleep in a day.
- **Target**:
  - `health_score`: A numeric value measuring health on a scale (e.g., approximately between 15 and 100).

You can use a real-world dataset (e.g., from Kaggle) or generate synthetic data using the methods included in the code.

### **Synthetic Dataset Note**
The synthetic dataset is based on simple assumptions, such as proportional relationships between steps, calories, and sleep, with added random noise. While useful for demonstrating the project, synthetic data may not reflect the complexity of real-world data. For better performance, consider using a real dataset.

---

## **Usage**

[//]: # (### 1. Train the Model)

[//]: # (To train the neural network, run:)

[//]: # (```bash)

[//]: # (python src/train.py)

[//]: # (```)

[//]: # (This script processes the dataset, trains the model, and saves it to the `models/` directory.)

[//]: # ()
[//]: # (### 2. Evaluate the Model)

[//]: # (To evaluate the trained model on a test dataset, run:)

[//]: # (```bash)

[//]: # (python src/evaluate.py)

[//]: # (```)

### Run the Dash Dashboard
Before running the application, set up the environment by executing:
```bash
source setup.sh
```

To use the interactive dashboard for predictions, run:
```bash
python -m dashboard/app.py
```
Once running, open your browser and navigate to:

#### Input Ranges
For accurate predictions, ensure your inputs match the following ranges based on the training dataset:
- `steps`: 1,000 to 20,000
- `calories_burned`: 1,200 to 4,500
- `sleep_hours`: 4 to 12

Predictions might be unreliable if the input values are significantly outside these ranges.

---

## **Dash Application**
- The Dash application is a simple web-based interface for entering input metrics like steps, calories burned, and sleep hours.
- The model predicts a health score that is displayed in real time.
- No input validation is performed directly in the dashboard, so use valid ranges as described above when entering metrics.

---

## **Model Details**
- **Architecture**:
  - Input layer with 3 features (`steps`, `calories_burned`, `sleep_hours`).
  - Three hidden layers with 64, 32, and 16 neurons (using ReLU activation).
  - Output layer with 1 neuron for the health score prediction.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam (Learning rate = 0.001).
- **Evaluation Metrics**: MSE, R² Score.

---

## **Model Saving and Loading**
- The trained model is saved to the `models/` directory in the `.pth` format.
- A `scaler.pkl` file containing the scaler for data normalization is also saved in the same directory.
- To reuse the trained model, you can load it as follows:
  ```python
  import torch
  model = torch.load('models/health_score_model.pth')
  model.eval()
  ```

---

## **Example Predictions**
Below are sample inputs and the corresponding health scores predicted by the model:

| Steps  | Calories Burned | Sleep Hours | Predicted Health Score |
|--------|------------------|-------------|-------------------------|
| 6,000  | 2,200            | 7           | 72.5                   |
| 15,000 | 3,800            | 8.5         | 89.2                   |
| 3,000  | 1,500            | 6           | 45.3                   |

---

## **Future Enhancements**
- Support for additional features like heart rate or distance traveled.
- Integration of real-world datasets for more accurate predictions.
- Visualization features in the dashboard, such as trend charts.
- Advanced neural network architectures for time-series data.

---

## **License**
This project is licensed under the MIT License. Feel free to use, modify, and distribute as needed.

---

## **Acknowledgments**
- [PyTorch](https://pytorch.org/) for enabling neural network implementation.
- [Dash](https://dash.plotly.com/) for building the dashboard interface.
- Kaggle datasets for inspiration and synthetic data ideas.