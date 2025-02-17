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
pip install torch pandas numpy scikit-learn dash matplotlib
```

---

## **Dataset**
The dataset consists of daily activity metrics:
- **Inputs**:
  - `steps`: Number of steps taken in a day.
  - `calories_burned`: Calories burned during the day.
  - `sleep_hours`: Total hours of sleep in a day.
- **Target**:
  - `health_score`: A numeric value measuring health on a scale (e.g., between 50 and 100).

You can use a real-world dataset from sources like Kaggle or generate a synthetic dataset included in the code.

---

## **Project Structure**
```plaintext
health-predictor/
├── datasets/                # (Optional) Folder for datasets
├── models/                  # Saved NN models
├── dashboard/               # Dash application code
├── src/                     # Source code for training and prediction
│   ├── dataset.py           # Data loading and preprocessing
│   ├── nn_model.py          # PyTorch neural network implementation
│   ├── train.py             # Model training script
│   ├── evaluate.py          # Evaluation script
├── app.py                   # Main entry point for the Dash app
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
```

---

## **Usage**

### 1. Train the Model
To train the neural network, run:
```bash
python src/train.py
```
This will preprocess the dataset, train the model, and save it to the `models/` directory.

### 2. Evaluate the Model
To evaluate the trained model on the test dataset, run:
```bash
python src/evaluate.py
```

### 3. Run the Dash Dashboard
To launch the interactive dashboard for predictions:
```bash
python app.py
```
Once running, go to `http://127.0.0.1:8050/` in your web browser to access the application.

---

## **Dash Application**
- The Dash application provides an easy-to-use interface for entering metrics such as steps, calories burned, and sleep hours.
- The model predicts a corresponding health score and displays it in real time.

---

## **Model Details**
- **Architecture**:
  - Input layer with 3 features (`steps`, `calories_burned`, `sleep_hours`).
  - Two hidden layers with 64 and 32 neurons, ReLU activation.
  - Output layer with 1 neuron for the health score prediction.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: Adam (Learning rate = 0.001).
- **Evaluation Metrics**: MSE, R² Score.

---

## **Future Enhancements**
- Support for additional health metrics (e.g., heart rate, distance traveled).
- Integrate real-world datasets for better predictions.
- Add visualization components in the Dash dashboard (e.g., trend charts).
- Experiment with advanced neural network architectures, e.g., deeper networks or recurrent models for time-series data.

---

## **License**
This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## **Acknowledgments**
- [PyTorch](https://pytorch.org/) for the neural network framework.
- [Dash](https://dash.plotly.com/) for the dashboarding capabilities.
- Kaggle for health-related datasets that inspired the project.