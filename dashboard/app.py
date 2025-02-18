import dash
from dash import html, dcc, Input, Output
import torch
import joblib
from models.nn_model import HealthScoreNN

def predict_health_score(steps, calories, sleep):
    data = [[steps, calories, sleep]]
    scaler = joblib.load('models/scaler.pkl')
    scaled_data = scaler.transform(data)
    input_data = torch.tensor(scaled_data, dtype=torch.float32)
    model = torch.load('models/health_score_model.pth', weights_only=False)
    res = model(input_data)
    return res.item()

app = dash.Dash()

app.layout = html.Div([
    html.H1("Health Score Predictor"),
    dcc.Input(id='input-steps', type='number', placeholder='Steps'),
    dcc.Input(id='input-calories', type='number', placeholder='Calories Burned'),
    dcc.Input(id='input-sleep', type='number', placeholder='Sleep Hours'),
    html.Button('Predict', id='predict-button'),
    html.H2(id='prediction-output')
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('input-steps', 'value'),
     Input('input-calories', 'value'),
     Input('input-sleep', 'value')]
)
def update_prediction(steps, calories, sleep):
    if None in [steps, calories, sleep]:
        return "Please enter all inputs"
    prediction = predict_health_score(steps, calories, sleep)
    return f"Predicted Health Score: {prediction:.1f}"


if __name__ == '__main__':
    app.run_server(debug=True)
