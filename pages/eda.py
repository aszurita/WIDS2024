from dash import Dash, dcc, html
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

training_df  = pd.read_csv("assets/data/training.csv")
training_df['DiagPeriodL90D'] = training_df['DiagPeriodL90D'].map({1: "Si", 0: "No"})
corre_df=pd.read_csv("assets/data/correlaciones.csv")
df_states=pd.read_csv('assets/data/df_states.csv')
df_race=pd.read_csv('assets/data/df_Race.csv')
subTitulo=html.Div(html.H3("Correlaciones".title(),className='subtitutlo-analisis'),className='left-align')
object_cols = list(training_df.select_dtypes(include='object').columns)

def armar_scatter(df_corre, label, df):
    df_corre_filtered = df_corre[df_corre['source'] == label]
    x_values = df_corre_filtered['source']
    y_values = df_corre_filtered['target']
    custom_colors = ['green','orange','red', 'green', 'blue']
    listaScatt = [px.scatter(df, x=x_value, y=y_value, color_discrete_sequence=[color]) for x_value, y_value, color in zip(x_values, y_values, custom_colors)]
    return listaScatt

labelcorre=''

div_graficorre=html.Div([
        html.Div([
            html.Label('Feature',className='labels'),
            dcc.Dropdown(corre_df['source'].unique(), id='dropdown1Corre',className='dropdown-feature',value='population'),
        ]), html.Br(),
    ],className='center')

sct2_1=html.Div([
        html.Div([
            dcc.Graph(id='fig_sca1')
        ],className='rounded-graph'),
        html.Div([
            dcc.Graph(id='fig_sca2')
        ],className='rounded-graph')
    ], className='center_row_around')

sct2_2=html.Div([
        html.Div([
            dcc.Graph(id='fig_sca3')
        ],className='rounded-graph'),
        html.Div([
            dcc.Graph(id='fig_sca4')
        ],className='rounded-graph')
    ], className='center_row_around', style={'padding-top': '60px'})
@app.callback(
    [
    Output('fig_sca1', 'figure'),
    Output('fig_sca2', 'figure'),
    Output('fig_sca3', 'figure'),
    Output('fig_sca4', 'figure')],
    [Input('dropdown1Corre', 'value')])
def update(dropdown1Corre):
    global labelcorre
    labelcorre=dropdown1Corre
    list_scatt=armar_scatter(corre_df,dropdown1Corre,training_df)
    return list_scatt[0],list_scatt[1],list_scatt[2],list_scatt[3]

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
    fig.update_layout( title=titulo)
    return fig

analisi_corre=html.Div([html.Br(),
        html.Label('Mayor Correlación ',className='labels'), 
        html.Div([
            html.Br(),
            html.Div(id='colMayorCorr',  className='labels'),
            html.Br(),
            dcc.Dropdown(object_cols, id='drdw2Corre',className='dropdown-feature',value=object_cols[0]),
            html.Br()
        ])])

pilas= html.Div([
            html.Div([html.Div([
                dcc.Graph(id='fig_posi',config={'responsive': True}),
            ],className='another-container')],className='centered-container2'),
            html.Div([html.Div([
                dcc.Graph(id='fig_nega',config={'responsive': True}),
            ],className='another-container')],className='centered-container2')
        ],className='center')

@app.callback(
    [Output('colMayorCorr', 'children'),
    Output('fig_posi', 'figure'),
    Output('fig_nega', 'figure')],
    [Input('drdw2Corre','value')])

def update_graph(drdw2Corre):
    titleMayorCor=maxCorre(corre_df,labelcorre)
    sinOutlier(training_df, titleMayorCor[0]),sinOutlier(training_df, titleMayorCor[1])
    df_positivo = training_df[training_df['DiagPeriodL90D'] == 'Si']
    df_negativo = training_df[training_df['DiagPeriodL90D'] == 'No']
    titMayorCor=f'{titleMayorCor[0].title()} - {titleMayorCor[1].title()}: {titleMayorCor[2]}'
    return titMayorCor,gfScaBox(df_positivo,titleMayorCor[0],titleMayorCor[1],drdw2Corre,'Diagnostico Positivo'),gfScaBox(df_negativo,titleMayorCor[0],titleMayorCor[1],drdw2Corre,'Diagnostico Negativo')

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
        #width=800, 
        #height=500 
    )
    return fig

grfMapa=html.Div([
    html.Div([ html.Div([
                dcc.Graph(figure=grfMap(df_states))
            ],className='rounded-graph')  ],className='centered-container')
        ],className='center')

def top_states(df_state_value,stateData):
    top10_state_yes = df_state_value.sort_values(stateData,ascending=False)
    top10_yesdiag_bar = px.bar(
                    top10_state_yes[:10],x='State',y=stateData,color='State',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title='TOP 10 States with the highest percentage of being diagnosed before 90 days'.title(),
                    labels={stateData:'Percentage'},
                    text_auto=True,hover_data={'Diagnostic90D':True})
    return top10_yesdiag_bar

sub_tiuloMap=html.Div(html.H3("Top de los Estados".title(),className='subtitutlo-analisis'),className='left-align')

top_state=html.Div([
        html.Div([
            dcc.RadioItems(
            id='rd_topSta',
            options=['Positivo','Negativo'],
            value='Positivo',  
            #labelStyle={'padding': '10px', 'margin-right': '10px', 'display': 'inline-block'}
            )
            ])])
grafTop=html.Div([html.Div([
            dcc.Graph(id='grf_top')
        ],className='rounded-graph') ],className='centered-container3')   
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
    #fig_barrmode_bmi_state.update_layout(height=1700, width=850)
    fig_barrmode_bmi_state.update_traces(width=0.7)
    return fig_barrmode_bmi_state

sub_tituBmi=html.Div(html.H3("Visulización de media de bmi por estado".title(),className='subtitutlo-analisis'),className='left-align')

state_bmi=html.Div([
        html.Div([
            dcc.Graph(figure=grf_bmiState(bmi_state(training_df)),config={'responsive': True})
        ],className='rounded-graph')    
    ],className='centered-container')

def top_PromedioParti(df_state_value,training_df):
    df_top_meanPM25 = df_state_value.sort_values(by='meanPM25', ascending=False)
    df_top_meanPM25=df_top_meanPM25.head(5)
    df_filtradoPorTOp=training_df[training_df['patient_state'].isin(df_top_meanPM25['State'])]
    df_filtradoPorTOp_posi=df_filtradoPorTOp[df_filtradoPorTOp['DiagPeriodL90D'] == 'Si']
    df_filtradoPorTOp_nega=df_filtradoPorTOp[df_filtradoPorTOp['DiagPeriodL90D'] == 'No']
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
    #fig.update_layout(width=800,height=600,title_text='Dispersion de particulas agrupados por estados')
    return fig

titu_tiParti=html.H1("Analisis por Particulas ",className='title_Corre')

df_parti=top_PromedioParti(df_states,training_df)

state_parti=html.Div([
        html.Div([
            dcc.Graph(figure=grf_boxPart(df_parti[0],df_parti[1]))
        ],className='rounded-graph')    
    ],className='centered-container')

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
            html.Div([dcc.Graph(id='fig_scatter',config={'responsive': True})],className='rounded-graph')
            ],className='centered-container')  
    ],className='center ')

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
            dcc.Graph(figure=fig_race(df_race),config={'responsive': True})
        ],className='rounded-graph')    
    ],className='centered-container')

layout = html.Div([
     html.Link(href='/assets/css/Eda.css', rel='stylesheet'),
   subTitulo,div_graficorre,sct2_1,sct2_2,analisi_corre,pilas,tiuloMap,grfMapa,sub_tiuloMap,top_state,grafTop,
     sub_tituBmi,state_bmi,titu_tiParti,state_parti,sub_tituParti,labels_parti,titu_Race,reace_fig
],className='body_model')