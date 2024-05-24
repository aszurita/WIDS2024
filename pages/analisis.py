from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from summarytools import dfSummary
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML
from app import app
html.Link(rel='stylesheet', href='/assets/all_event.css')
training_df  = pd.read_csv("assets/data/training.csv")
training_df['DiagPeriodL90D'] = training_df['DiagPeriodL90D'].map({1: "Si", 0: "No"})
corre_df=pd.read_csv("assets/data/correlaciones.csv")
df_encoder = pd.read_csv('assets/data/df_encoder.csv')
df_states=pd.read_csv('assets/data/df_states.csv')
df_race=pd.read_csv('assets/data/df_Race.csv')

links = html.Div([
    html.A(html.Label('Datos',className='link'),href='#datos'),
    html.A(html.Label('Correlaciones',className='link'),href='#correlacion'),
    html.A(html.Label('Distribuciones',className='link'),href='#distribucion'),
    html.A(html.Label('Seccion4',className='link'),href='#id4'),
],className='row_head')

titulo  =  html.H1("ANÁLISIS EXPLORATORIO DE DATOS".title(),className='titutlo-analisis')

def div_feature(feature):
    return html.P(feature,className='div_feature')

object_cols = list(training_df.select_dtypes(include='object').columns)
number_cols = list(training_df.select_dtypes(exclude='object').columns)

def div_listfeatures(type,list_features):
    className_extra = ' '
    if type == 'Categóricas' : 
        className_extra = 'Categoricas'
    else : 
        className_extra = 'Numericas'
    htmls = [ div_feature(col) for col in list_features]
    result = html.Div([
        html.Label(type,className='label_features'),
        html.Div(htmls,className='features')
    ],className=f'div_features {className_extra}')
    return result

def bar_tipoDatos():
    df_tipodatos = pd.DataFrame({
        'Tipo' : ['Object','Number'],
        'Values':[len(object_cols),len(number_cols)]
    })
    df_tipodatos['Porcentaje'] = (df_tipodatos['Values'] / len(training_df.columns)).round(2)
    fig = px.bar(df_tipodatos, x='Tipo', y='Values' ,color='Tipo',
        title='TIPOS DE DATOS',height=455,color_discrete_sequence = px.colors.qualitative.Pastel,
        text_auto =True, hover_name='Tipo', hover_data={'Porcentaje':True,'Tipo':False})
    return fig


tipo_datos = html.Div([
    html.H3('Tipos De Datos',className='title_Corre'),
    html.Div([
        dcc.Graph(figure=bar_tipoDatos(),className="rounded-graph2")
    ],className='center_figure'),
    html.Div([
        div_listfeatures('Categóricas',object_cols),
        div_listfeatures('Numéricas',number_cols),
    ],className='div_tiposdatos')
],className='center_div',id='datos')

summary = dfSummary(training_df)
summary_html = summary.to_html()

fast_analisis = html.Div([
    html.H1("Análisis Rapido",className='title_Corre'),
    html.H1("Podemos visualizar un análisis rápido de todas las características, como valores estadísticos, valores únicos, valores faltantes, etc.",className='texto'),
    html.Div(DangerouslySetInnerHTML(summary_html),className='resumen')
],className='div_resumen_general')

fast_analisis2 = html.Div([
    html.A([
        html.Label("Visualizar análisis rapido de los datos", className="label_eda"),
        html.Img(src="assets/images/image.png", className="imagen_eda")
    ], href="https://angelofasteda.000webhostapp.com/", target="_blank", className="div_imagen")
],className="div_fastAnalisis2")



div_correlation = html.Div([
    html.H3('Correlaciones',className='title_Corre'),
    dcc.Dropdown(
                id='column-corr-f1',
                options=[{'label': col, 'value': col} for col in training_df.columns],
                className='dropdown-feature'
            ),
    html.P('Top 10 con mayor correlación :',className='subtitle'),
    html.Div(id='table_correlacion',className="div_table")
    ],className='div_correlation',id='correlacion')


@app.callback(
    Output('table_correlacion','children'),
    Input('column-corr-f1','value'),
)
def table_correlacion(col): 
    if col != None:
        correlaciones = df_encoder.corr()[col].reset_index()
        correlaciones.columns = ['Features','Correlación']
        correlaciones['Correlación'] = abs(correlaciones['Correlación']).round(4)
        correlaciones = correlaciones.sort_values('Correlación',ascending=False)
        return  dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in correlaciones.columns],
            data=correlaciones.to_dict('records'),
            style_table={'height': '350px', 'overflowY': 'auto',},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'center',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'minWidth': '350px', 'width': '350px', 'maxWidth': '400px',
            })
    return html.Div()



div_graficas_features = html.Div([
    html.H3('Distribución De Datos',className='title_Corre2'),
    html.Div(dbc.Button('Graficar',id='button-graficar',className='button-graficar')),
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
                    options=[{'label': col, 'value': col} for col in (training_df.columns.to_list()+['frecuencia'])],
                    className='dropdown-feature'
                ),
            html.Div(id='histograma-feature3')
        ],className='center_col feature'),
        html.Div(id='div_scatter',className='center_col feature div_resultado'),
    ],className='w-all center_row_around'),
],className='center_col gap_30 mt-30 body_graficas w-all',id='distribucion')

def histogram(col1):
    fig = px.histogram(training_df, x=col1 , marginal='box',
                        color_discrete_sequence = px.colors.qualitative.Pastel,
                        title=f"'{col1.upper()}' ",
                        labels={col1:col1.upper()})
    mean = training_df[col1].mean()
    fig.add_vline(x=mean, line_width=3, line_dash='dash', line_color='rgb(254,136,177)', annotation_text=f' Mean: {mean:.2f}',
                                    annotation_font_size=14, annotation_font_color='rgb(254,136,177)')
    return fig

def bar(col1):
    value_counts = training_df[col1].value_counts().reset_index()
    value_counts.columns = [col1,'Value']
    fig = px.bar(value_counts,x=col1,y='Value', color=col1,
                title=f"'{col1.upper()}'",text_auto=True,
                color_discrete_sequence = px.colors.qualitative.Pastel,
                labels={col1:col1.upper()})
    return fig

def scatter(col1,col2,color=None):
    col1_ = col1 or ''
    col2_ = col2 or ''
    color_label = color or ''
    fig = px.scatter(training_df, x=col1 , y=col2 ,color=color,
                            color_discrete_sequence = px.colors.qualitative.Pastel, color_continuous_scale='tealgrn',
                            title=f"{col1_.upper()} VS {col2_.upper()}",height=455,
                            labels={col1:col1_.upper(),col2:col2_.upper(),color:color_label.upper()})
    return fig

def scatter_frecuencia(col1,col2,color):
    if col1 and col2 and col1!=col2 : 
        df = training_df.groupby([col1,col2])[[col1,col2]].value_counts().reset_index()
        df.columns=[col1,col2,'frecuencia']
    elif col1 and not col2 :
        df = training_df.groupby([col1])[[col1]].value_counts().reset_index()
        df.columns=[col1,'frecuencia']
    else :
        df = training_df.groupby([col2])[[col2]].value_counts().reset_index()
        df.columns=[col2,'frecuencia']
    col1_ = col1 or ''
    col2_ = col2 or ''
    color_label = color or ''
    fig = px.scatter(df, x=col1 , y=col2 ,color=color,
                            color_discrete_sequence = px.colors.qualitative.Pastel,
                            color_continuous_scale='tealgrn',size=color,
                            title=f"{col1_.upper()} VS {col2_.upper()}",height=455,
                            labels={col1:col1_.upper(),col2:col2_.upper(),color:color_label.upper()})
    return fig

def div_graficar_col(col1):
    if not col1:
        return html.Div() 
    if col1 == 'frecuencia' :
        return html.Div(html.Img(src='assets/images/Frecuencia.png',width=705,height=453)) 
    if training_df[col1].dtype == 'object':
        graph = dcc.Graph(figure=bar(col1),className="graph_res")
    else:
        graph = dcc.Graph(figure=histogram(col1),className="graph_res")
    return graph

def div_scatter(col1, col2, color):

    if not col1 and not col2:
        return html.Div()
    
    if color == 'frecuencia':
        function = scatter_frecuencia
    else:
        function = scatter
    
    primary_col = col1 if col1 else col2
    secondary_col = col2 if col1 else col1
    graph = dcc.Graph(figure=function(primary_col, secondary_col, color),className="graph_res")
    label = html.Label('Resultado', className='labels')
    return [label, graph]

@app.callback(
    [
        Output('histograma-feature1','children'),
        Output('histograma-feature2','children'),
        Output('histograma-feature3','children'),
        Output('div_scatter','children'),
    ],
    [
        Input('button-graficar','n_clicks'),
        Input('column-dropdown1','value'),
        Input('column-dropdown2','value'),
        Input('column-dropdown3','value') 
    ],
)
def graficar_scatter(n_clicks,col1,col2,color):
    fig1,fig2,fig3,fig4 = html.Div(),html.Div(),html.Div(),html.Div()
    fig1 = div_graficar_col(col1)
    fig2 = div_graficar_col(col2)
    fig3 = div_graficar_col(color)
    if n_clicks != None and  n_clicks > 0:
        fig4 = div_scatter(col1,col2,color)  
    return fig1,fig2,fig3,fig4


layout = html.Div([
        html.Link(rel='stylesheet', href='/assets/Analisis.css'),
        links,titulo,tipo_datos,fast_analisis,fast_analisis2,div_correlation,div_graficas_features]
    ,className='center body'
    )