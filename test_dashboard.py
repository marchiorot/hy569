import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
# import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import numpy as np

# from demo import make_prediction, load_model

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div([
        html.H4('Stress live updates'),
        html.Div(id='live-update-text'),
        #dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=7*1000, # in milliseconds
            n_intervals=0
        )
    ])
)

# CHANGE THIS
def obtain_values():
    user_id = 100
    with open('stress.txt','r') as f:
        status = f.readlines()
    stress = status
    with open('probs.txt','r') as f:
        probs = f.readlines()
    confidence = probs
    return user_id, stress[0], confidence[0]


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    
    # print('here')
    user_id, stress, confidence = obtain_values()
    # print(user_id)
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('ID: {}'.format(user_id), style=style),
        html.Span('Stress: {}'.format(stress), style=style),
        html.Span('Confidence: {}'.format(confidence), style=style)
    ]

app.run_server(debug=True)
