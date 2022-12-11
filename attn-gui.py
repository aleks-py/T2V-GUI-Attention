from dash import Dash, dcc, html, Input, Output, State, ctx
from plotly.subplots import make_subplots
from dash.dependencies import Input, Output
import dash_daq as daq
from dash import html
# from jupyter_dash import JupyterDash
from dash import dcc
# import torch
import numpy as np
import plotly.graph_objs as go
import pickle
import urllib.request

app = Dash(__name__)
server = app.server

all_attn = pickle.load(
    urllib.request.urlopen('https://drive.google.com/uc?export=download&id=1iu5K3w_rm6W7iZ_NqX0mYQvwVefYjudo'))
# all_attn = torch.load('C:\\Users\\alexs\\Dropbox (MIT)\\Subjects_Fall2022\\6.S898\\Project\\Code\\Frey_Faces_VAE\\frey_faces_attention_matrix.pt')
orig = pickle.load(
    urllib.request.urlopen('https://drive.google.com/uc?export=download&id=1BIm81rVdzlb-AnXk0pZQki7zLBnzgRWT'))
# orig = torch.load('C:\\Users\\alexs\\Dropbox (MIT)\\Subjects_Fall2022\\6.S898\\Project\\Code\\Frey_Faces_VAE\\frey_original_for_attention.pt')
max_locs = 60
pad = 6
locs = np.linspace(pad, max_locs - pad, 5).astype(int)

attn = dcc.Graph(id="attention", config={'staticPlot': True},
                 style={'float': 'center', 'margin': 'auto'}
                 )

app.layout = html.Div([
    html.Div(attn, style={'position': 'absolute', 'margin-left': 20, 'margin-top': -40}),
    html.H1('Original Image',
            style={'position': 'absolute', 'color': '#b5b5b5', 'fontSize': 18, 'margin-left': 110, 'margin-top': 30,
                   'font-family': 'Arial'}),
    html.H1('Attention Map',
            style={'position': 'absolute', 'color': '#b5b5b5', 'fontSize': 18, 'margin-left': 300, 'margin-top': 30,
                   'font-family': 'Arial'}),

    html.Button(id='x0y0', n_clicks=0, style={'position': 'absolute', 'margin-left': 100, 'margin-top': 60,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x0y1', n_clicks=0, style={'position': 'absolute', 'margin-left': 100, 'margin-top': 104,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x0y2', n_clicks=0, style={'position': 'absolute', 'margin-left': 100, 'margin-top': 148,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x0y3', n_clicks=0, style={'position': 'absolute', 'margin-left': 100, 'margin-top': 192,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x0y4', n_clicks=0, style={'position': 'absolute', 'margin-left': 100, 'margin-top': 236,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '44px', 'width': '32px'}),

    html.Button(id='x1y0', n_clicks=0, style={'position': 'absolute', 'margin-left': 131, 'margin-top': 60,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x1y1', n_clicks=0, style={'position': 'absolute', 'margin-left': 131, 'margin-top': 104,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x1y2', n_clicks=0, style={'position': 'absolute', 'margin-left': 131, 'margin-top': 148,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x1y3', n_clicks=0, style={'position': 'absolute', 'margin-left': 131, 'margin-top': 192,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x1y4', n_clicks=0, style={'position': 'absolute', 'margin-left': 131, 'margin-top': 236,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '44px', 'width': '31px'}),

    html.Button(id='x2y0', n_clicks=0, style={'position': 'absolute', 'margin-left': 161, 'margin-top': 60,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x2y1', n_clicks=0, style={'position': 'absolute', 'margin-left': 161, 'margin-top': 104,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x2y2', n_clicks=0, style={'position': 'absolute', 'margin-left': 161, 'margin-top': 148,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x2y3', n_clicks=0, style={'position': 'absolute', 'margin-left': 161, 'margin-top': 192,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x2y4', n_clicks=0, style={'position': 'absolute', 'margin-left': 161, 'margin-top': 236,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '44px', 'width': '31px'}),

    html.Button(id='x3y0', n_clicks=0, style={'position': 'absolute', 'margin-left': 191, 'margin-top': 60,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x3y1', n_clicks=0, style={'position': 'absolute', 'margin-left': 191, 'margin-top': 104,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x3y2', n_clicks=0, style={'position': 'absolute', 'margin-left': 191, 'margin-top': 148,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x3y3', n_clicks=0, style={'position': 'absolute', 'margin-left': 191, 'margin-top': 192,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '31px'}),
    html.Button(id='x3y4', n_clicks=0, style={'position': 'absolute', 'margin-left': 191, 'margin-top': 236,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '44px', 'width': '31px'}),

    html.Button(id='x4y0', n_clicks=0, style={'position': 'absolute', 'margin-left': 221, 'margin-top': 60,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x4y1', n_clicks=0, style={'position': 'absolute', 'margin-left': 221, 'margin-top': 104,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x4y2', n_clicks=0, style={'position': 'absolute', 'margin-left': 221, 'margin-top': 148,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x4y3', n_clicks=0, style={'position': 'absolute', 'margin-left': 221, 'margin-top': 192,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '45px', 'width': '32px'}),
    html.Button(id='x4y4', n_clicks=0, style={'position': 'absolute', 'margin-left': 221, 'margin-top': 236,
                                              "backgroundColor": "transparent", 'border': '1px black dotted',
                                              'height': '44px', 'width': '32px'}),

])


@app.callback(
    Output('attention', 'figure'),
    Input('x0y0', 'n_clicks'),
    Input('x0y1', 'n_clicks'),
    Input('x0y2', 'n_clicks'),
    Input('x0y3', 'n_clicks'),
    Input('x0y4', 'n_clicks'),

    Input('x1y0', 'n_clicks'),
    Input('x1y1', 'n_clicks'),
    Input('x1y2', 'n_clicks'),
    Input('x1y3', 'n_clicks'),
    Input('x1y4', 'n_clicks'),

    Input('x2y0', 'n_clicks'),
    Input('x2y1', 'n_clicks'),
    Input('x2y2', 'n_clicks'),
    Input('x2y3', 'n_clicks'),
    Input('x2y4', 'n_clicks'),

    Input('x3y0', 'n_clicks'),
    Input('x3y1', 'n_clicks'),
    Input('x3y2', 'n_clicks'),
    Input('x3y3', 'n_clicks'),
    Input('x3y4', 'n_clicks'),

    Input('x4y0', 'n_clicks'),
    Input('x4y1', 'n_clicks'),
    Input('x4y2', 'n_clicks'),
    Input('x4y3', 'n_clicks'),
    Input('x4y4', 'n_clicks'),
)
def update_output(b1, b2, b3, b4, b5,
                  b6, b7, b8, b9, b10,
                  b11, b12, b13, b14, b15,
                  b16, b17, b18, b19, b20,
                  b21, b22, b23, b24, b25):
    x = None
    y = None
    # x=0
    if "x0y0" == ctx.triggered_id:
        x = 0
        y = 0
    elif "x0y1" == ctx.triggered_id:
        x = 0
        y = 1
    elif "x0y2" == ctx.triggered_id:
        x = 0
        y = 2
    elif "x0y3" == ctx.triggered_id:
        x = 0
        y = 3
    elif "x0y4" == ctx.triggered_id:
        x = 0
        y = 4
    # x=1
    elif "x1y0" == ctx.triggered_id:
        x = 1
        y = 0
    elif "x1y1" == ctx.triggered_id:
        x = 1
        y = 1
    elif "x1y2" == ctx.triggered_id:
        x = 1
        y = 2
    elif "x1y3" == ctx.triggered_id:
        x = 1
        y = 3
    elif "x1y4" == ctx.triggered_id:
        x = 1
        y = 4
    # x=2
    elif "x2y0" == ctx.triggered_id:
        x = 2
        y = 0
    elif "x2y1" == ctx.triggered_id:
        x = 2
        y = 1
    elif "x2y2" == ctx.triggered_id:
        x = 2
        y = 2
    elif "x2y3" == ctx.triggered_id:
        x = 2
        y = 3
    elif "x2y4" == ctx.triggered_id:
        x = 2
        y = 4
    # x=3
    elif "x3y0" == ctx.triggered_id:
        x = 3
        y = 0
    elif "x3y1" == ctx.triggered_id:
        x = 3
        y = 1
    elif "x3y2" == ctx.triggered_id:
        x = 3
        y = 2
    elif "x3y3" == ctx.triggered_id:
        x = 3
        y = 3
    elif "x3y4" == ctx.triggered_id:
        x = 3
        y = 4
    # x=4
    elif "x4y0" == ctx.triggered_id:
        x = 4
        y = 0
    elif "x4y1" == ctx.triggered_id:
        x = 4
        y = 1
    elif "x4y2" == ctx.triggered_id:
        x = 4
        y = 2
    elif "x4y3" == ctx.triggered_id:
        x = 4
        y = 3
    elif "x4y4" == ctx.triggered_id:
        x = 4
        y = 4

    if x == None or y == None:
        img_attn = np.zeros(orig.shape)
        img_attn[:, :] = np.nan
    else:
        img_attn = all_attn[x, y][0]

    layout = go.Layout(
        height=400,
        width=500,
    )

    figure = make_subplots(rows=1, cols=2)
    # original image
    figure.add_trace(go.Heatmap(z=orig, colorscale='gray', colorbar=None, showlegend=False, hoverinfo='none'),
                     row=1, col=1)
    # attention map
    figure.add_trace(go.Heatmap(z=img_attn, colorscale='viridis', colorbar=None, showlegend=False, hoverinfo='none'),
                     row=1, col=2)
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_traces(showscale=False)
    figure.update_layout(layout)
    # scatter point
    if x == None or y == None:
        pass
    else:
        x_coord, y_coord = all_attn[x, y][1]

        figure.add_trace(
            go.Scatter(x=[x_coord], y=[420 - y_coord], marker=dict(size=10, color='red', symbol='square', opacity=0.6)),
            row=1, col=1)
        figure.add_trace(
            go.Scatter(x=[x_coord], y=[420 - y_coord], marker=dict(size=10, color='red', symbol='square', opacity=0.6)),
            row=1, col=2)
    figure.update_traces(showlegend=False)
    return figure


if __name__ == '__main__':
    app.run_server(debug=False)