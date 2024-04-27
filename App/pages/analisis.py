from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from app import app
import plotly.graph_objects as go
from plotly.subplots import make_subplots

titulo  =  html.H1("ANÁLISIS EXPLORATORIO DE DATOS".title(),className='titutlo-analisis')


training_df  = pd.read_csv("assets/data/training.csv")
corre_df=pd.read_csv("assets/data/correlaciones.csv")
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
        html.Label('Agrupar' ,htmlFor='column-dropdown3',className='labels'),
        dcc.Dropdown(
                id='column-dropdown3',
                options=[{'label': col, 'value': col} for col in training_df.columns],
                className='dropdown-feature'
            ),
        html.Div(id='histograma-feature3')
    ],className='center_col feature')
],className='center_col gap_20')

scatter_general = html.Div(id='div_scatter')

div_grafica = html.Div([div_graficas_features,scatter_general],className='center_col')


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
            fig = dcc.Graph(figure=bar(col1))
        else :
            fig =  dcc.Graph(figure=histogram(col1))
    else : 
        fig  = html.Div()
    return fig

def div_scatter(col1,col2,color):
    fig= ''
    if col1 or col2 : 
        if col1 : 
            fig = [dcc.Graph(figure=scatter(col1,col2,color))]
        else:
            fig = [dcc.Graph(figure=scatter(col2,col1,color))]
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

subTitulo=html.Div(html.H2("Correlaciones".title(),className='subtitutlo-analisis'),className='left-align')

def armar_scatter(df_corre,label,df):
  fig = make_subplots(rows=2, cols=2)
  df_corre = df_corre[df_corre['source']==label]
  for x in range(2):
    for y in range(2):
      if x==0 and 0 <= y < df_corre.shape[0]:
        fig.add_trace(go.Scatter(x=df[label], y=df[df_corre.iloc[y]['target']],mode='markers',name=df_corre.iloc[y]['value']),row=x+1, col=y+1)
        fig.update_xaxes(title_text=label, row=x+1, col=y+1)
        fig.update_yaxes(title_text=df_corre.iloc[y]['target'], row=x+1, col=y+1)
      elif 2 <= y+2 < df_corre.shape[0]:
        fig.add_trace(go.Scatter(x=df[label], y=df[df_corre.iloc[y+2]['target']],mode='markers',name=df_corre.iloc[y+2]['value']),row=x+1, col=y+1)
        fig.update_xaxes(title_text=label, row=x+1, col=y+1)
        fig.update_yaxes(title_text=df_corre.iloc[y+2]['target'], row=x+1, col=y+1)
      else: continue
  fig.update_layout( title_text=f'Mapas de distribucion',width=1000,height=800,showlegend=True)
  return fig

div_graficorre=html.Div([
        html.Div([
            html.Label('Feature',className='labels'),
            dcc.Dropdown(corre_df['source'].unique(), id='dropdown1Corre',className='dropdown-feature',value='population'),
        ]),
        html.Div([
            dcc.Graph(id='figs_graficas')])
    ],className='center')
def spliDataCorre(value):
    return value
    
analisi_corre=html.Div([
        html.Div(id='mayorCorrela')
    ])
@app.callback(
    [Output('mayorCorrela', 'children'),
    Output('figs_graficas', 'figure')],
    Input('dropdown1Corre', 'value'))

def update_graph(value):
    return value,armar_scatter(corre_df,value,training_df)

layout = html.Div(
    [titulo,div_grafica,button,subTitulo,div_graficorre,analisi_corre]
    ,className='center body'
    )   