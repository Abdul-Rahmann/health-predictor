import dash
from dash import html, dcc, Input, Output
import numpy as np


# Dummy model predictions (replace with your model)
def predict_health_score(steps, calories, sleep):
    # Pass inputs through the trained model instead
    return 10 + steps * 0.01 + calories * 0.1 + sleep * 1.5


# Dash app setup
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
    return f"Predicted Health Score: {prediction:.2f}"


if __name__ == '__main__':
    app.run_server(debug=True)
