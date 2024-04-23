from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from app import app

titulo  =  html.H1("ANÁLISIS EXPLORATORIO DE DATOS".title(),className='titutlo-analisis')


training_df  = pd.read_csv("assets/data/training.csv")
# Div Dos Drowpdown y scatter plot que representa la correlación
div_graficas_features = html.Div([
    html.Div([
        html.Div([
            html.Label('Feature 1',htmlFor='column-dropdown1',className='labels'),
            dcc.Dropdown(
                    id='column-dropdown1',
                    options=[{'label': col, 'value': col} for col in training_df.columns],
                    className='dropdown-feature'
                ),
            html.Div(id='histograma-feature1')
        ],className='center_col feature'),
        html.Div([
            html.Label('Feature 2',htmlFor='column-dropdown2',className='labels'),
            dcc.Dropdown(
                    id='column-dropdown2',
                    options=[{'label': col, 'value': col} for col in training_df.columns],
                    className='dropdown-feature'
                ),
            html.Div(id='histograma-feature2')
        ],className='center_col feature')
    ],className='w-all center_row_around'),
    html.Div([
        html.Div([
            html.Label('Agrupar' ,htmlFor='column-dropdown3',className='labels'),
            dcc.Dropdown(
                    id='column-dropdown3',
                    options=[{'label': col, 'value': col} for col in training_df.columns],
                    className='dropdown-feature'
                ),
            html.Div(id='histograma-feature3')
        ],className='center_col feature'),
        html.Div(id='div_scatter',className='center_col feature'),
    ],className='w-all center_row_around'),
],className='center_col gap_30 mt-30')



button = html.Div(dbc.Button('Graficar',id='button-graficar',className='button-graficar'))

def scatter(col1,col2,color=None):
    col1_ = col1 or ''
    col2_ = col2 or ''
    color_label = color or ''
    fig = px.scatter(training_df, x=col1 , y=col2 ,color=color,
                            color_discrete_sequence = px.colors.qualitative.Plotly,
                            title=f"{col1_.upper()} VS {col2_.upper()}",
                            labels={col1:col1_.upper(),col2:col2_.upper(),color:color_label.upper()})
    return fig
def histogram(col1):
    fig = px.histogram(training_df, x=col1 , marginal='box',
                        color_discrete_sequence = px.colors.qualitative.Pastel,
                        title=f"DISTRIBUCIÓN DE DATOS DE LA COLUMNA '{col1.upper()}' ", width=700,
                        labels={col1:col1.upper()})
    mean = training_df[col1].mean()
    fig.add_vline(x=mean, line_width=3, line_dash='dash', line_color='rgb(254,136,177)', annotation_text=f' Mean: {mean:.2f}',
                                    annotation_font_size=14, annotation_font_color='rgb(254,136,177)')
    return fig

def bar(col1):
    value_counts = training_df[col1].value_counts().reset_index()
    value_counts.columns = [col1,'Value']
    fig = px.bar(value_counts,x=col1,y='Value', color=col1,
                title=f"DISTRIBUCIÓN DE DATOS DE LA COLUMNA '{col1.upper()}'",text_auto=True,
                color_discrete_sequence = px.colors.qualitative.Pastel,
                labels={col1:col1.upper()})
    return fig


def div_graficar_col(col1):
    fig  = ''
    if col1 : 
        if training_df[col1].dtype == object : 
            fig = [dcc.Graph(figure=bar(col1))]
        else :
            fig =  dcc.Graph(figure=histogram(col1))
    else : 
        fig  = html.Div()
    return fig

def div_scatter(col1,col2,color):
    fig= ''
    if col1 or col2 : 
        if col1 : 
            fig = [html.Label('Resultado',className='labels'),dcc.Graph(figure=scatter(col1,col2,color))]
        else:
            fig = [html.Label('Resultado',className='labels'),dcc.Graph(figure=scatter(col2,col1,color))]
    else : 
        fig  = html.Div()
    return fig


@app.callback(
    [
        Output('histograma-feature1','children'),
        Output('histograma-feature2','children'),
        Output('histograma-feature3','children'),
        Output('div_scatter','children'),
    ],
    Input('button-graficar','n_clicks'),
    [
        State('column-dropdown1','value'),
        State('column-dropdown2','value'),
        State('column-dropdown3','value')
    ],

)
def graficar_scatter(n_clicks,col1,col2,color):
    fig1,fig2,fig3,fig4 = html.Div(),html.Div(),html.Div(),html.Div()
    if n_clicks != None and  n_clicks > 0:
        fig1 = div_graficar_col(col1)
        fig2 = div_graficar_col(col2)
        fig3 = div_graficar_col(color)
        fig4 = div_scatter(col1,col2,color)  
    return fig1,fig2,fig3,fig4

layout = html.Div(
    [titulo,div_graficas_features,button]
    ,className='center body'
    )