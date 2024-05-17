from dash import dcc, html
from dash.dependencies import Input, Output
from app import app
from pages import analisis, modelo, home, eda

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Img(src='/assets/images/Logo_wids2024.png', className='logo'),
        html.Nav([
            dcc.Link('Inicio', href='/home'),
            dcc.Link('Análisis', href='/analisis'),
            dcc.Link('Modelo', href='/modelo'),
            dcc.Link('Eda', href='/eda'),
            
        ], className='navbar'),
    ], className='div_nav'),
    html.Div(id='page-content', className='div_carrusel')
], className= 'div_general')


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/analisis':
        return analisis.layout
    elif pathname == '/modelo':
        return modelo.layout
    elif pathname == '/eda':
        return eda.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)
