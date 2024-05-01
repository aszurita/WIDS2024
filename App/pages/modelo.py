from dash import Dash, dcc, html,dash_table
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import json
from app import app
import os
import pickle

with open('assets/data/datos.json', 'r') as file:
    datos_models = json.load(file)


title = html.H1('Modelo',className='title_modelo')


def button(id,titulo):
    return dbc.Button(titulo,id=id,className='button_Model')

lista_butons = [button(info_model['id'],info_model['titulo']) for info_model in datos_models ]

row_buttons = html.Div(lista_butons,className='butons')


def encontrar_titulo_por_id(id_buscado):
    for modelo in datos_models:
        if modelo['id'] == id_buscado:
            return modelo['titulo']
    return None

def encontrar_modelo_por_id(id_buscado):
    for modelo in datos_models:
        if modelo['id'] == id_buscado:
            return modelo
    return None



lista_info = [
    ["TN (True Negative)","No había cáncer y el modelo predijo correctamente que no había cáncer."],
    ["FP (False Positive)","No había cáncer pero el modelo predijo erróneamente que había cáncer"],
    ["FN (False Negative)","Había cáncer pero el modelo predijo erróneamente que no había cáncer."],
    ["TP (True Positive)","Había cáncer y el modelo predijo correctamente que había cáncer."],
]
def div_color(id , info):
    return html.Div([
        html.Div(id=id,className='w-h-100'),
        html.Div([
            html.P(info[0],className='info_color'),
            html.P(info[1],className=''),
        ],className='col_info')
    ],className='row_color')

color_confusion_matrix = html.Div([
    div_color('TN_C' , lista_info[0]),
    div_color('FP_C' , lista_info[1]),
    div_color('FN_C' , lista_info[2]),
    div_color('TP_C' , lista_info[3])
],className='row_colors')

titulo_matrix = html.H3('Matriz De Confusión',className='bold')
confusion_matrix = html.Div([
    html.Div([
        html.Div(id='TN', className='value_matrix'),
        html.Div(id='FP', className='value_matrix')
    ], className='row_matrix'),
    html.Div([
        html.Div(id='FN', className='value_matrix'),
        html.Div(id='TP', className='value_matrix')
    ], className='row_matrix')
], className='matrix_confusion')

div_confusion_matrix = html.Div([
    titulo_matrix,
    html.Div([
        color_confusion_matrix,
        confusion_matrix
    ],className='row_confu_matrix')
],className='div_confusion_matrix')

parametros = html.Div([
    html.Div([
        html.H2('Parametros',className='label_para'),
        html.Div(id='imagen',className='img_center'),
    ],className='col imagen_pa'),
    html.Div([
        html.H2('Accuracy',className='label_para'),
        html.Div(id='accuracy',className='text_accuracy')
    ],className='col')
],className='div_parametros')

div_infoModel = html.Div([  
    html.Div(id='label_Model'),
    div_confusion_matrix,
    parametros
],className='div_infomodels')

div_models = html.Div([
    row_buttons,div_infoModel
],className='div_models')

def label_value(value):
    return html.Label(value,className='valuenumber_matrix')

@app.callback(
    [
        Output('label_Model','children'),
        Output('TN','children'),
        Output('FP','children'),
        Output('FN','children'),
        Output('TP','children'),
        Output('imagen','children'),
        Output('accuracy','children'),
    ],
    [
        Input(datos_models[0]['id'],'n_clicks'),
        Input(datos_models[1]['id'],'n_clicks'),
        Input(datos_models[2]['id'],'n_clicks')
    ]
)
def update_div_model(button1,button2,button3):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    label = html.Label(encontrar_titulo_por_id(button_id),className='labelModel')
    TN,FP,FN,TP,imagen,accuracy = html.Div(),html.Div(),html.Div(),html.Div(),html.Div(),html.Div()
    button_id = button_id or 'Modelo1' # Predeterminado
    if button_id:
        modelo  = encontrar_modelo_por_id(button_id)
        lista_values = modelo['confusion_matrix']
        TN,FP,FN,TP = label_value(lista_values[0]),label_value(lista_values[1]),label_value(lista_values[2]),label_value(lista_values[3])
        imagen = html.Img(src=modelo['parametros'],className='imagen')
        accuracy = html.P(round(modelo['accuracy'],4))
    return label,TN,FP,FN,TP,imagen,accuracy


training_df  = pd.read_csv("assets/data/training.csv")



def input_dropdown(label):
    values  = list(training_df[label].sort_values().unique())
    no_hay = True   
    for value in values:
        if pd.isna(value):
            no_hay = False
    if no_hay:
        values.append(pd.NA)
    dropdown = dcc.Dropdown(
                options=[{'label': 'Otro', 'value': f'{pd.NA}'} if pd.isna(x) else {'label': f'{x}', 'value': f'{x}'} for x in values],
                id=label,
                className='dropdown_inputs',
                searchable=True,
            )
    dropdown_descr = dcc.Dropdown(
                options=[{'label': 'Otro', 'value': f'{pd.NA}'} if pd.isna(x) else {'label': f'{x}', 'value': f'{x}'} for x in values],
                id=label,
                className='dropdown_inputs_Desc',
                searchable=True,
            )
    input_float = dcc.Input(
                id=columnas_input[6], 
                type='number',
                min=1,  
                max=90, 
                step=0.01,  
                placeholder='BMI',
                className='dropdown_inputs'
                )
    input_number = dcc.Input(
                id=columnas_input[3], 
                type='number',
                min=100,  
                max=999, 
                step=1,  
                placeholder='ZIP',
                className='dropdown_inputs'
                )
    input_number_age=dcc.Input(
        id=columnas_input[4], 
        type='number',
        min=15,  
        max=120, 
        step=1,  
        placeholder='ZIP',
        className='dropdown_inputs'
    )
    classname = 'div_dropdown'
    if label == columnas_input[6]:
        input = input_float
    elif label == columnas_input[3]:
        input = input_number
    elif label == columnas_input[4]:
        input = input_number_age
    elif label == columnas_input[-1]:
        input = dropdown_descr
        classname='div_dropdown_desc'
    else : 
        input = dropdown


    div = html.Div([
        html.Label(label,className='label_dropdown'),
        input
    ],className=classname)
    return div


button_model = html.Div([
    dbc.Button('Predecir',id='button_predecir',className='button_predict')
],className='div_dropdown')



columnas_input = ['patient_race', 'payer_type', 'patient_state',
    'patient_zip3', 'patient_age', 'patient_gender', 'bmi',
    'breast_cancer_diagnosis_code','metastatic_cancer_diagnosis_code', 'Region', 'Division','breast_cancer_diagnosis_desc']

inputs = html.Div([
    html.Div([
        input_dropdown(columnas_input[0]),
        input_dropdown(columnas_input[1]),
        input_dropdown(columnas_input[2]),
        input_dropdown(columnas_input[3]),
    ],className='row_inputs'),
    html.Div([
        input_dropdown(columnas_input[4]),
        input_dropdown(columnas_input[5]),
        input_dropdown(columnas_input[6]),
        input_dropdown(columnas_input[7]),
    ],className='row_inputs'),
    html.Div([
        input_dropdown(columnas_input[8]),
        input_dropdown(columnas_input[9]),
        input_dropdown(columnas_input[10]),
    ],className='row_inputs'),
    html.Div([
        input_dropdown(columnas_input[11]),
        button_model
    ],className='row_inputs'),
],className='col_inputs')



div_best_model  = html.Div([
    html.Label("Mejor Modelo",className='labelModel'),
    html.Div([
        html.Label(encontrar_modelo_por_id('Modelo3')['titulo'],className='label_best_model'),
        html.H2('Accuracy',className='label_acc_best'),
        html.Div(round(encontrar_modelo_por_id('Modelo3')['accuracy'],4),id='accuracy_thebest',className='best_acc')
    ],className='card_info'),
    inputs,
    html.Div([
        html.Div(id='table_features',className='col_table'),
        html.Div(id='resultados',className='col_table')
    ],className='row_result')
],className='div_models')



zip3_df = pd.read_csv('assets/data/patient_zip3.csv')
zip3_df.set_index('patient_zip3',inplace=True)

all_columns = columnas_input+list(zip3_df.columns)+['metastatic_first_novel_treatment','metastatic_first_novel_treatment_type']

object_cols = list(training_df.select_dtypes(include='object').columns)
number_cols = list(training_df.select_dtypes(exclude='object').columns)
number_cols.remove('DiagPeriodL90D')
number_cols.remove('patient_id')

CatBoostClassifier_path = 'models/CatBoostClassifier.sav'
with open(CatBoostClassifier_path, 'rb') as model_file:
    catBoostClassifier = pickle.load(model_file)
Encoder_data = os.path.join('models/Encoder.sav')
with open(Encoder_data, 'rb') as model_file:
    encoder= pickle.load(model_file)



@app.callback(
    [Output('table_features','children'),
    Output('resultados','children')
    ],
    [State(col,'value') for col in columnas_input],
    Input('button_predecir','n_clicks')
)
def inputs_predict(*args) :
    div_table,div_resul = html.Div(),html.Div()
    if args[-1] != None and  args[-1] > 0 : 
        lista_values_zip3 = [np.nan]*len(zip3_df.columns)
        if(args[3] in list(training_df['patient_zip3'].unique())):
            lista_values_zip3 = zip3_df.loc[args[3]].to_list()
        lista_values = [args[0] or np.nan,
                        args[1] or np.nan,
                        args[2] or np.nan,
                        args[3] or np.nan,
                        args[4] or np.nan,
                        args[5] or np.nan,
                        args[6] or np.nan,
                        args[7] or np.nan,
                        args[8] or np.nan,
                        args[9] or np.nan,
                        args[10] or np.nan,
                        args[11] or np.nan,
                        ]
        all_values = lista_values+lista_values_zip3+[np.nan]*2
        df_predict = pd.DataFrame(columns=all_columns)
        df_predict.loc[0]  = all_values
        transpuesto = df_predict.T.reset_index()
        transpuesto.columns = ['Features','Values']
        transpuesto = transpuesto.round(4)
        title = html.Label('Valores',className='label_table')
        table_features  = dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in transpuesto.columns],
        data=transpuesto.to_dict('records'),
        style_table={'height': '350px', 'overflowY': 'auto'},
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_cell={
            'textAlign': 'center',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'minWidth': '300px', 'width': '180px', 'maxWidth': '180px',
        })
        div_table = [title,table_features]
        df_predict_objects_col = encoder.transform(df_predict[object_cols])
        df_predict_encoder = pd.concat([df_predict_objects_col.reset_index(drop=True), df_predict[number_cols].reset_index(drop=True)], axis=1)
        y_proba = catBoostClassifier.predict_proba(df_predict_encoder)
        prob_1 = np.round(y_proba[0, 1],3)
        prob_0 = np.round(y_proba[0, 0],3)
        data = pd.DataFrame({
            'Categoria': ['SI DIAGNOSTICADO 90D', 'NO DIAGNOSTICADO 90D'],
            'Probabilidad': [prob_1, prob_0]
        })
        fig_probabilidad = px.pie(data, values='Probabilidad', names='Categoria', title='Probabilidad de ser diagnosticado dentro de los 90 días'.title(),color='Categoria',
                                color_discrete_map={'BENIGN':'rgb(26, 77, 128)','MALIGNANT':'rgb(128, 26, 26)'},width=500)
        title_result= html.Label('Resultado',className='label_table')
        figrure = dcc.Graph(figure=fig_probabilidad)
        div_resul = [title_result,figrure]
    return div_table,div_resul

layout = html.Div([
    title, div_models,div_best_model
],className='body_model')