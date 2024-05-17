from dash import Dash, dcc, html,dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output, ALL
from app import app

df = pd.read_csv("assets/data/colum.csv", sep="|")

carousel_items = [
    {"key": "1", "src": "/assets/images/wids1.jpg", "title": "Dathaton", "text": "WIDS Espol 2024"},
    {"key": "2", "src": "/assets/images/wids2.jpg", "title": "Datathon WiDS Espol", "text": "Es una competencia de ciencia de datos que ofrece a los estudiantes la oportunidad de sumergirse en el análisis de datos. Busca inspirar a la próxima generación de científicos de datos y promover un cambio en la resolución de problemas mediante el uso de datos."},
    {"key": "3", "src": "/assets/images/wids3.jpeg", "title": "Temas a Tratar", "text": "-Predicción Temprana, Factores Demográficos y Ambientales, Accesibilidad al Tratamiento, e Innovaciones en Análisis de Datos"},
    {"key": "4", "src": "/assets/images/wids4.jpg", "title": "Únete y usa tus skills ", "text": "de data para luchar contra el cáncer más rápido."},
]

indicators = html.Div([
    html.Span(id={'type': 'carousel-indicator', 'index': idx}, className='carousel-indicator')
    for idx in range(len(carousel_items))
], className='carousel-indicators')

infoDat = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H2('Descripción del conjunto de datos:'),
                        html.Ul([
                            html.Li('El WiDS Datathon 2024 se centra en una tarea de predicción utilizando un conjunto de datos compuesto por aproximadamente 18,000 registros, que representan a pacientes con cáncer de mama metastásico triple negativo en los Estados Unidos.'),
                            html.Li('Este evento es organizado por Women in Data Science (WiDS) Worldwide, que tiene como misión aumentar la participación de las mujeres en la ciencia de datos para beneficiar a las sociedades en todo el mundo.'),
                            html.Li('Los datos incluyen una amplia gama de características sobre los pacientes, así como información relevante sobre su diagnóstico, tratamiento y contexto socioeconómico y geográfico.'),
                            html.Li('Cada fila del conjunto de datos corresponde a un único paciente y su período de diagnóstico.'),
                        ]),
                    ],
                    className='info-box',style={'margin-top': '20px'}
                ),
                html.Div(
                    [
                        html.H2('Origen de los datos:'),
                        html.Ul([
                            html.Li('El conjunto de datos utilizado en este desafío proviene de Health Verity (HV), uno de los mayores ecosistemas de datos de atención médica en los Estados Unidos.'),
                            html.Li('Health Verity proporciona datos relacionados con la salud de pacientes diagnosticados con cáncer de mama metastásico triple negativo.'),
                            html.Li('Estos datos se han enriquecido con una amplia base de códigos postales de EE. UU., así como datos socioeconómicos y de salud a nivel de código postal, incluyendo información sobre la calidad del aire y su relación con la salud.'),
                        ]),
                    ],
                    className='info-box'
                ),
                html.Div(
                    [
                        html.H2('Tarea del desafío:'),
                        html.P('La tarea consiste en evaluar si la probabilidad de que el período de diagnóstico del paciente sea inferior a 90 días es predecible utilizando estas características e información sobre el paciente.'),
                    ],
                    className='info-box'
                ),
                html.Div(
                    [
                        html.H2('Los Labels con los datos:'),
                        html.Label("Desliza hacia la derecha para explorar el significado",className="text-appear")
                    ],
                    className='info-box labels-box'
                ),
            ],
            className='info-container'
        )
    ],className="container_info"
)

tablaColum=html.Div(
    className='table-container',
    children=[
        html.Div(
            className='table-wrapper',  
            children=[
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'id': c, 'name': c} for c in df.columns],
                    style_cell={'textAlign': 'left', 'whiteSpace': 'normal'},  
                    style_data={
                        'color': 'black',
                        'backgroundColor': 'white'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(223, 242, 254)',
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(173, 164, 255)', 
                        'color': 'white',
                        'fontWeight': 'bold',
                    },
                    style_table={'overflowX': 'auto'}  
                ),
            ]
        )
    ]
)

layout = html.Div([
    dcc.Interval(id='interval-component', interval=8*1000, n_intervals=0),
    html.Div(id='carousel-content', className='carousel'),
    indicators, infoDat,tablaColum
], className='div_general')

@app.callback(
    [Output('carousel-content', 'children'),
     Output({'type': 'carousel-indicator', 'index': ALL}, 'className')],
    [Input('interval-component', 'n_intervals'),
     Input({'type': 'carousel-indicator', 'index': ALL}, 'n_clicks')]
)
def update_carousel(n_intervals, n_clicks):
    # Determine the index of the current item
    if any(n_clicks):
        idx = n_clicks.index(1)
    else:
        idx = n_intervals % len(carousel_items)
    
    item = carousel_items[idx]

    carousel = html.Div([
        html.Div([
            html.H1(item["title"], style={'margin': '0'}, className='h1_texto'),
            html.P(item["text"], style={'margin': '0'}, className= 'p_texto')
        ], className='carousel-text'),
        html.Img(src=item["src"])
        
        
    ], className='carousel')

    indicators_class = ['carousel-indicator'] * len(carousel_items)
    indicators_class[idx] += ' active'

    return carousel, indicators_class
