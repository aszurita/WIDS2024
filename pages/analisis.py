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
        dcc.Graph(figure=bar_tipoDatos(),className="rounded-graph")
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

final = html.Div([],className='final')
subTitulo=html.Div(html.H3("Correlaciones".title(),className='subtitutlo-analisis'),className='left-align')

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
  fig.update_layout( title_text=f'Mapas de distribucion',width=1000,height=700,showlegend=True)
  return fig

div_graficorre=html.Div([
        html.Div([
            html.Label('Feature',className='labels'),
            dcc.Dropdown(corre_df['source'].unique(), id='dropdown1Corre',className='dropdown-feature',value='population'),
        ]), html.Br(),
        html.Div([
            dcc.Graph(id='figs_graficas')],className='rounded-graph')
    ],className='center ')

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

analisi_corre=html.Div([html.Br(),
        html.Label('Mayor Correlación ',className='labels'), 
        html.Div([
            html.Br(),
            html.Div(id='colMayorCorr',  className='labels'),
            html.Br(),
            dcc.Dropdown(object_cols, id='drdw2Corre',className='dropdown-feature',value=object_cols[0]),
            html.Br()
        ]),
        html.Div([
            html.Div([
                dcc.Graph(id='fig_posi'),
            ],className='rounded-graph'),
            html.Div([
                dcc.Graph(id='fig_nega'),
            ],className='rounded-graph')
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

tiuloMap=html.H1("Analisis por Estados ",className='title_Corre')

def grfMap(df_state_value):
    fig = px.choropleth(
        df_state_value,
        locations='State',
        locationmode='USA-states',
        color='Value',
        hover_name='State',
        hover_data={'State': False, 'Value': True,'Diagnostic90D':True,'NoDiagnostic90D':True},
        color_continuous_scale='Sunset',
        scope='usa',
        title='Los pacientes se distribuyen en los sgts Estados'
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        coloraxis_colorbar=dict(
            title='Cantidad',
            tickvals=[i for i in range(0, int(df_state_value['Value'].max())+1, 250)]
        ),
        width=800, 
        height=500 
    )
    return fig

grfMapa=html.Div([
            html.Div([
                dcc.Graph(figure=grfMap(df_states))
            ],className='rounded-graph')   
        ],className='center')

def top_states(df_state_value,stateData):
    top10_state_yes = df_state_value.sort_values(stateData,ascending=False)
    top10_yesdiag_bar = px.bar(
                    top10_state_yes[:10],x='State',y=stateData,color='State',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='TOP 10 States with the highest percentage of being diagnosed before 90 days'.title(),
                    labels={stateData:'Percentage'},
                    text_auto=True,hover_data={'Diagnostic90D':True},width=850,height=500 )
    return top10_yesdiag_bar

sub_tiuloMap=html.Div(html.H3("Top de los Estados".title(),className='subtitutlo-analisis'),className='left-align')

top_state=html.Div([
        html.Div([
            dcc.RadioItems(
            id='rd_topSta',
            options=['Positivo','Negativo'],
            value='Positivo',  
            labelStyle={'padding': '10px', 'margin-right': '10px', 'display': 'inline-block'})]),
        html.Div([
            dcc.Graph(id='grf_top')
        ],className='rounded-graph')    
    ])
@app.callback(
    Output('grf_top','figure'),
    Input('rd_topSta','value')
)
def grf_topCa(rd_topSta):
    if rd_topSta=='Positivo':
        return top_states(df_states,'StateDiagnostic90DPercentage')
    return top_states(df_states,'StateNoDiagnostic90DPercentage')

def bmi_state(training_df):
    bmi_without_null = training_df.dropna(subset='bmi')
    bmi_state = bmi_without_null.groupby('patient_state').agg(
     mean_bmi=pd.NamedAgg(column='bmi', aggfunc='mean'),).reset_index()
    bmi_state.columns=['State','Mean_Bmi']
    return bmi_state

def grf_bmiState(df_bmi):
    fig_barrmode_bmi_state = px.bar(df_bmi.sort_values('Mean_Bmi',ascending=False), y='State' , x ='Mean_Bmi',barmode = 'group',
                                 labels={'value':'Median BMI','variable':''},text_auto=True,orientation='h',
                                 color_discrete_sequence=px.colors.qualitative.Pastel,color = 'State',
                                  hover_name='State',hover_data={'State': False}, title='Bmi By State'.title() )
    fig_barrmode_bmi_state.update_layout(height=1700, width=850)
    fig_barrmode_bmi_state.update_traces(width=0.7)
    return fig_barrmode_bmi_state

sub_tituBmi=html.Div(html.H3("Visulización de media de bmi por estado".title(),className='subtitutlo-analisis'),className='left-align')

state_bmi=html.Div([
        html.Div([
            dcc.Graph(figure=grf_bmiState(bmi_state(training_df)))
        ],className='rounded-graph')    
    ])

def top_PromedioParti(df_state_value,training_df):
    df_top_meanPM25 = df_state_value.sort_values(by='meanPM25', ascending=False)
    df_top_meanPM25=df_top_meanPM25.head(5)
    df_filtradoPorTOp=training_df[training_df['patient_state'].isin(df_top_meanPM25['State'])]
    df_filtradoPorTOp_posi=df_filtradoPorTOp[df_filtradoPorTOp['DiagPeriodL90D'] == 1.]
    df_filtradoPorTOp_nega=df_filtradoPorTOp[df_filtradoPorTOp['DiagPeriodL90D'] == 0.]
    return [df_filtradoPorTOp_posi,df_filtradoPorTOp_nega]

def grf_boxPart(df_filtradoPorTOp_posi,df_filtradoPorTOp_nega):
    lista_colors=['blue','red','green','rgb(93, 197, 244 )','rgb(248, 115, 89 )','rgb(40, 226, 99 )']
    lista_nombres=['PM25 Positivo','N02 Positivo','Ozone Positivo','PM25 Negativo','N02 Negativo','Ozone Negativo']
    lista_labe=['PM25','N02','Ozone']
    fig = make_subplots(rows=2, cols=3)
    for x in range(2):
        for y in range(3):
            if x==1:
                fig.add_trace(go.Box(x=df_filtradoPorTOp_posi['patient_state'], y=df_filtradoPorTOp_posi[lista_labe[y]],name=lista_nombres[y], marker_color=lista_colors[y]),row=x+1, col=y+1)
            else:
                fig.add_trace(go.Box(x=df_filtradoPorTOp_nega['patient_state'], y=df_filtradoPorTOp_nega[lista_labe[y]],name=lista_nombres[y+3], marker_color=lista_colors[y+3]),row=x+1, col=y+1)
    fig.update_traces(quartilemethod="exclusive")
    fig.update_layout(width=800,height=600,title_text='Dispersion de particulas agrupados por estados')
    return fig

titu_tiParti=html.H1("Analisis por Particulas ",className='title_Corre')

df_parti=top_PromedioParti(df_states,training_df)

state_parti=html.Div([
        html.Div([
            dcc.Graph(figure=grf_boxPart(df_parti[0],df_parti[1]))
        ],className='rounded-graph')    
    ])

def scatter_parti(df_state_value,label):
    fig = px.scatter(df_state_value, x=label, y='StateDiagnostic90DPercentage',color='State',
                 labels={
                     label: f'Niveles de {label}',
                     'StateDiagnostic90DPercentage': 'Tasa de incidencia de cáncer'
                 },
                 size='StateDiagnostic90DPercentage',
                 size_max=20 ,
                 title=f'Relación entre niveles de {label} y tasas de incidencia de cáncer')
    fig.update_layout(width=1000)
    return fig

sub_tituParti=html.Div(html.H3("Relacion entre los niveles de particulas y Incidencia de deteccion".title(),className='subtitutlo-analisis'),className='left-align')

labels_parti=html.Div([
        html.Div([
            html.Label('Tipo de particula',className=''),
            dcc.Dropdown(options=[
                {'label': 'NO2', 'value': 'meanN02'},
                {'label': 'PM25', 'value': 'meanPM25'},
                {'label': 'Ozono', 'value': 'meanOzone'},
   ], id='ddw_parti',className='dropdown-feature',value='meanN02'),
        ]), html.Br(),
        html.Div([
            dcc.Graph(id='fig_scatter')],className='rounded-graph')
    ],className='center')

@app.callback(
    Output('fig_scatter', 'figure'),
    Input('ddw_parti', 'value')
)
def drop_part(value):
    return scatter_parti(df_states,value)

def fig_race(df_race):
    bar_race = px.bar(
    df_race.sort_values('Percentage', ascending=False),
    y=['Yes_Diag_PercentageTotal', 'No_Diag_PercentageTotal'], x='patient_race', title="Distribution of Diagnosis by Race",
    labels={'patient_race': 'patient_race'.title(), 'value': 'Percentage', 'variable': 'Diagnosis Type','Yes_Diag_PercentageTotal':'Yes_DiagPeriodL90D',
            'No_Diag_PercentageTotal':'No_DiagPeriodL90D'},
    text_auto=True,
    color_discrete_sequence=px.colors.qualitative.Pastel,
    hover_data=['Total_Count'],
    barmode='group',
    )
    for trace in bar_race.data:
        if trace.name == 'Yes_Diag_PercentageTotal':
            trace.name = 'YesDiagPeriodL90D'
        elif trace.name == 'No_Diag_PercentageTotal':
            trace.name = 'NoDiagPeriodL90D'
    return bar_race

titu_Race=html.H1("Analisis por Raza",className='title_Corre')
reace_fig=html.Div([
        html.Div([
            dcc.Graph(figure=fig_race(df_race))
        ],className='rounded-graph')    
    ])
# layout = html.Div(
#     [links,titulo,tipo_datos,fast_analisis,fast_analisis2,div_correlation,div_graficas_features,
#     final,subTitulo,div_graficorre,analisi_corre,tiuloMap,grfMapa,sub_tiuloMap,top_state,
#     sub_tituBmi,state_bmi,titu_tiParti,state_parti,sub_tituParti,labels_parti,titu_Race,reace_fig]
#     ,className='center body'
#     )

layout = html.Div([
        html.Link(rel='stylesheet', href='/assets/Analisis.css'),
        links,titulo,tipo_datos,fast_analisis,fast_analisis2,div_correlation,div_graficas_features]
    ,className='center body'
    )