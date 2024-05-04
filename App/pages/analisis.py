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
links = html.Div([
    html.A(html.Label('Datos',className='link'),href='#datos'),
    html.A(html.Label('Correlaciones',className='link'),href='#correlacion'),
    html.A(html.Label('Distribuciones',className='link'),href='#distribucion'),
    html.A(html.Label('Seccion4',className='link'),href='#id4'),
],className='row_header')

titulo  =  html.H1("ANÁLISIS EXPLORATORIO DE DATOS".title(),className='titutlo-analisis')


training_df  = pd.read_csv("assets/data/training.csv")
corre_df=pd.read_csv("assets/data/correlaciones.csv")
df_encoder = pd.read_csv('assets/data/df_encoder.csv')

def div_feature(feature):
    return html.Div(feature,className='div_feature')

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
        dcc.Graph(figure=bar_tipoDatos())
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


div_correlation = html.Div([
    html.H3('Correlaciones',className='title_Corre'),
    dcc.Dropdown(
                id='column-corr-f1',
                options=[{'label': col, 'value': col} for col in training_df.columns],
                className='dropdown-feature'
            ),
    html.P('Top 10 con mayor correlación :',className='subtitle'),
    html.Div(id='table_correlacion')
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
],className='center_col gap_30 mt-30 body_graficas',id='distribucion')





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
        graph = dcc.Graph(figure=bar(col1))
    else:
        graph = dcc.Graph(figure=histogram(col1))
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
    graph = dcc.Graph(figure=function(primary_col, secondary_col, color))
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

final = html.Div([],className='final')
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

def maxCorre(df,label):
    dfFil = df[df['source'] == label]
    idmax = dfFil['value'].abs().idxmax()
    rowdata = dfFil.loc[idmax]
    return rowdata

def sinOutlier(df, label):
    Q1 = df[label].quantile(0.25)
    Q3 = df[label].quantile(0.75)
    IQR = Q3 - Q1
    limiInfe = Q1 - 1.5 * IQR
    limiSupe = Q3 + 1.5 * IQR
    df_filtered = df[(df[label] >= limiInfe) & (df[label] <= limiSupe)].copy()
    df[:] = df_filtered
    df.reset_index(drop=True, inplace=True)

def gfScaBox(df, label1, label2, categ,titulo):
    fig = px.scatter(df, x=label1, y=label2, color=categ, marginal_y="box",
                     marginal_x="box")
    fig.update_layout(width=720, title=titulo,height=500)
    return fig

analisi_corre=html.Div([
        html.Label('Mayor Correlación ',className='labels'), 
        html.Div([
            html.Br(),
            html.Div(id='colMayorCorr',  className='labels'),
            html.Br(),
            dcc.Dropdown(['patient_race','payer_type','Region','Division'], id='drdw2Corre',className='dropdown-feature',value='patient_race'),
            html.Br()
        ]),
        html.Div([
            dcc.Graph(id='fig_posi'),
            dcc.Graph(id='fig_nega'),
        ],className='row_gr')
    ],className='center')
@app.callback(
    [Output('colMayorCorr', 'children'),
    Output('figs_graficas', 'figure'),
    Output('fig_posi', 'figure'),
    Output('fig_nega', 'figure')],
    [Input('dropdown1Corre', 'value'),
     Input('drdw2Corre','value')])

def update_graph(dropdown1Corre,drdw2Corre):
    titleMayorCor=maxCorre(corre_df,dropdown1Corre)
    sinOutlier(training_df, titleMayorCor[0]),sinOutlier(training_df, titleMayorCor[1])
    df_positivo = training_df[training_df['DiagPeriodL90D'] == 1.]
    df_negativo = training_df[training_df['DiagPeriodL90D'] == 0.]
    titMayorCor=f'{titleMayorCor[0].title()} - {titleMayorCor[1].title()}: {titleMayorCor[2]}'
    return titMayorCor,armar_scatter(corre_df,dropdown1Corre,training_df),gfScaBox(df_positivo,titleMayorCor[0],titleMayorCor[1],drdw2Corre,'Diagnostico Positivo'),gfScaBox(df_negativo,titleMayorCor[0],titleMayorCor[1],drdw2Corre,'Diagnostico Negativo')

layout = html.Div(
    [links,titulo,tipo_datos,fast_analisis,div_correlation,div_graficas_features,final,subTitulo,div_graficorre,analisi_corre]
    ,className='center body'
    )   