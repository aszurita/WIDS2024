from dash import Dash, dcc, html,dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

df = pd.read_csv("assets/data/colum.csv", sep="|")

# Componentes HTML y CSS
titleGeneral = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the ESPOL dashboard", className='titutlo-analisis')
                    , className="mb-4 mt-4")
        ]),
    ])
])

carousel = dbc.Carousel(
    items=[
        {
            "key": "1", 
            "src": "/assets/images/wids.png",
            "style": {
                "border-radius": "15px",
                "width": "200px", 
                "height": "100px",  
                "margin": "auto"  
            }
        },
    ],
    controls=False,
    indicators=False,
    interval=2000,
    ride="carousel",
    style={
        "border": "2px solid white",  
        "border-radius": "15px", 
        "overflow": "hidden" 
    }
)

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
                    ],
                    className='info-box labels-box'
                ),
            ],
            className='info-container'
        )
    ]
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
    titleGeneral,carousel,infoDat,tablaColum
],className='body_model')
