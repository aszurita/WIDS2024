import dash
import dash_bootstrap_components as dbc
import flask 
external_stylesheets = [dbc.themes.YETI]  

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

@server.route('/ModelNotebook/')
def serve_model_notebook():
    return flask.send_from_directory('assets/notebooks', 'Show_Models.html')

@server.route('/DataEncoderNotebook/')
def serve_data_encoder_notebook():
    return flask.send_from_directory('assets/notebooks', 'DataEncoder.html')
