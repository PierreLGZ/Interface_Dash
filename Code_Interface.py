# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



#%%Modélisation graphique
import base64
import io
import dash
from dash.dependencies import Input, Output, State
#import dash_core_components as dcc
from dash import dcc, callback_context
from dash import html
from dash import dash_table
import pandas as pd

#new imports----------------------
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from time import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.io as pio

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import plotly.graph_objects as go


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score

#--------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    dcc.Store(id="variable_cible_Stored",data=[],storage_type="local"),
    dcc.Store(id="variables_explis_Stored",data=[],storage_type="local"),
    dcc.Store(id="modele_ML_1_Stored",data=[],storage_type="local"),
    dcc.Store(id="modele_ML_2_Stored",data=[],storage_type="local"),
    dcc.Store(id="modele_ML_3_Stored",data=[],storage_type="local"),
    dcc.Store(id="data_Stored",data=[],storage_type="local"),
    dcc.Store(id="type_algo_Stored",data=[],storage_type="local"),
])

index_page = html.Div(
    id='index_page',
    children=[
        dcc.Upload(
            id='datatable-upload',
            children=html.Div([
                'Glisser et déposer ou ',
                html.A("Sélectionner un fichier CSV avec la virgule comme séparateur ")
            ]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
        ),
        dash_table.DataTable(id='datatable-upload-container'),
        html.Br(),
        #html.Div(id="id_data_informations"),
        #html.Br(),
        html.Div([
            html.H6("Sélectionner la variable à prédire : "),
            html.Div([
                dcc.Dropdown(
                    id='id_variable_cible',
                    options=[],
                    persistence=True, persistence_type='memory'
                    #,value='Sélectionner la variale à prédire'#ne sert peut etre à rien
            )], style={'width': '60%', 'display': 'inline-block'}),
            html.Br(),

            html.H6("Sélectionner la ou les variables explicatives ('All' pour toutes les variables) : "),
            dcc.Dropdown(
                id='id_var_expli',
                options=[],
                multi=True
            ),
            html.Br(),

            html.H6(id = "id_type_algos", children="Sélectionner le ou les algorithmes de Machine Learning pour effectuer l'apprentissage du modèle :"),
            #Les dropdowns --------------------------------
            html.Div([
                dcc.Dropdown(
                    id='id_Modele_ML1',
                    options=[],
                    persistence=True, persistence_type='memory'
                    #,value='Sélectionner le modèle' #ne sert peut etre à rien
                ),
            ], style={'width': '30%', 'display': 'inline-block'}),
            html.Div([
                dcc.Dropdown(
                    id='id_Modele_ML2',
                    options=[]
                    #,value='Sélectionner le modèle' #ne sert peut etre à rien
                ),
            ], style={'width': '30%', 'display': 'inline-block'}),
            html.Div([
                dcc.Dropdown(
                    id='id_Modele_ML3',
                    options=[]
                    #,value='Sélectionner le modèle' #ne sert peut etre à rien
                ),
            ], style={'width': '30%', 'display': 'inline-block'}),
            #--------------------------------

            #Les liens --------------------------------
            html.Div([
                html.Div([
                    #dcc.Link('RésultatsAlgo1', href='/page-1'),
                    html.A("RésultatsAlgo1", href='/page-1', target="_blank"),
                ], style={'display': 'inline-block',"margin-left": "10px"}),

                html.Div([
                    html.A("RésultatsAlgo2", href='/page-2', target="_blank"),
                ], style={'display': 'inline-block', "margin-left": "130px"}),
                html.Div([
                    html.A("RésultatsAlgo3", href='/page-3', target="_blank"),
                ], style={'display': 'inline-block', "margin-left": "130px"}),

            ], style={'width': '100%', 'display': 'inline-block'}),
            #--------------------------------


        ], style={'width': '48%', 'display': 'inline-block'}),
        #html.Button('Submit', id='submit_val', n_clicks=0),
        #dcc.Graph(id='results_graph_111'),
        #dcc.Graph(id='results_graph_222'),

        #html.Div(id= "test_on_submit_button"),


    ]
)


page_1_layout = html.Div(
    id = 'page_1_layout',
    children = [
            html.Div([
                html.Div([
                    html.H1(id="id_nom_algo1", children="Résultats pour l'algorithme N°1"),
                ], style={'width': '100%', 'display': 'inline-block', "text-align": "center","text-decoration": "underline"}),
                html.Br(),
                html.Div([
                    html.A("Résultats Algorithme N°2", href='/page-2', target="_blank", style={'color': 'dodgerblue','height': '50px','width': '300px'}),
                ], style={'display': 'inline-block', "margin-left": "1200px"}),

            ], style={'width': '100%', 'display': 'inline-block'}),
            html.Button('Afficher', id='submit_val_PAGE1', n_clicks=0, style={"margin-left": "100px",'background-color': 'aqua','color': 'black','height': '50px','width': '200px',"text-align": "center"}),
            html.Div(id='page-1-content'),
            html.Div(id="graph-container", children= [
                html.Br(),
                html.H6(id="id_res1_1"),
                html.Br(),
                html.H6(id="id_res2_1"),
                html.Br(),
                html.H6(id="id_res3_1"),
                html.Br(),
                html.H6(id="id_res4_1"),
                html.Br(),
                html.H6(id="id_res5_1"),
                html.Br(),
            ], style={"margin-left": "100px"}),

        #dcc.Link('Go to Page 2', href='/page-2'),
            #html.Br(),
            dcc.Graph(id='results_graph_1_PAGE1'),
            dcc.Graph(id='results_graph_2_PAGE1'),
            html.Div(id="test_on_submit_button_PAGE1"),
    ]
)

page_2_layout = html.Div(
    id='page_2_layout',
    children=[
        html.Div([
            html.Div([
                html.H1(id="id_nom_algo2", children="Résultats pour l'algorithme N°2"),
            ], style={'width': '100%', 'display': 'inline-block', "text-align": "center",
                      "text-decoration": "underline"}),
            html.Br(),
            html.Div([
                html.A("Résultats Algorithme N°3", href='/page-3', target="_blank",
                       style={'color': 'dodgerblue', 'height': '50px', 'width': '300px'}),
            ], style={'display': 'inline-block', "margin-left": "1200px"}),

        ], style={'width': '100%', 'display': 'inline-block'}),
        html.Button('Afficher', id='submit_val_PAGE2', n_clicks=0,
                    style={"margin-left": "100px", 'background-color': 'aqua', 'color': 'black', 'height': '50px',
                           'width': '200px', "text-align": "center"}),
        html.Div(id='page-2-content'),
        html.Div(id="graph-container", children=[
            html.Br(),
            html.H6(id="id_res1_2"),
            html.Br(),
            html.H6(id="id_res2_2"),
            html.Br(),
            html.H6(id="id_res3_2"),
            html.Br(),
            html.H6(id="id_res4_2"),
            html.Br(),
            html.H6(id="id_res5_2"),
            html.Br(),
        ], style={"margin-left": "100px"}),

        # dcc.Link('Go to Page 2', href='/page-2'),
        # html.Br(),
        dcc.Graph(id='results_graph_1_PAGE2'),
        dcc.Graph(id='results_graph_2_PAGE2'),
        html.Div(id="test_on_submit_button_PAGE2"),
    ]
)


page_3_layout = html.Div(
    id='page_3_layout',
    children=[
        html.Div([
            html.Div([
                html.H1(id="id_nom_algo3", children="Résultats pour l'algorithme N°3"),
            ], style={'width': '100%', 'display': 'inline-block', "text-align": "center",
                      "text-decoration": "underline"}),
            html.Br(),

        ], style={'width': '100%', 'display': 'inline-block'}),
        html.Button('Afficher', id='submit_val_PAGE3', n_clicks=0,
                    style={"margin-left": "100px", 'background-color': 'aqua', 'color': 'black', 'height': '50px',
                           'width': '200px', "text-align": "center"}),
        html.Div(id='page-3-content'),
        html.Div(id="graph-container", children=[
            html.Br(),
            html.H6(id="id_res1_3"),
            html.Br(),
            html.H6(id="id_res2_3"),
            html.Br(),
            html.H6(id="id_res3_3"),
            html.Br(),
            html.H6(id="id_res4_3"),
            html.Br(),
            html.H6(id="id_res5_3"),
            html.Br(),
        ], style={"margin-left": "100px"}),

        dcc.Graph(id='results_graph_1_PAGE3'),
        dcc.Graph(id='results_graph_2_PAGE3'),
        #html.Div(id="graph_container",
        #  children= [dcc.Graph(id='results_graph_2_PAGE3'),
        #]),

        html.Div(id="test_on_submit_button_PAGE3"),
    ]
)


#Callback pour afficher quelques lignes du dataframe, alimenter le nom des colonnes
#dans le tableau à afficher et enfin, alimenter la liste déroulante pour le choix de la variable cible
@app.callback(Output('datatable-upload-container', 'data'),
              Output('datatable-upload-container', 'columns'),
              Output('id_variable_cible', 'options'),
              Input('datatable-upload', 'contents'),
              State('datatable-upload', 'filename'))
def update_output(contents, filename):
    if contents is None:
        return [{}], []

    global df

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # On vérifie que ça soit un fichier csv
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Au dans le pire des cas un fichier excel
        df= pd.read_excel(io.BytesIO(decoded))
    #df = parse_contents(contents, filename)
    return df.head().to_dict('records'), [{"name": i, "id": i} for i in df.columns], [{"label": i, "value": i} for i in
                                                                               df.columns]

@app.callback(
    Output('id_var_expli', 'options'),
    Input('id_variable_cible', 'value'),
    State('datatable-upload', 'contents'))

def update_dropdown(id_var_cible,content):
    var_expli = df.columns
    var_expli2 = [var for var in var_expli if var != id_var_cible]
    return [{"label": "ALL", "value": "ALL"}]+[{"label": i, "value": i} for i in var_expli2]

@app.callback(
    Output('id_Modele_ML1', 'options'),
    Output('id_Modele_ML2', 'options'),
    Output('id_Modele_ML3', 'options'),
    Output('id_type_algos', 'children'),
    Input('id_variable_cible', 'value'),
    State('datatable-upload', 'contents'))

def set_good_Models(var_cible_name, contents):
    '''content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))'''
    if var_cible_name:
        if df[var_cible_name].dtype == 'object':
            models_dd_options = [{'label': 'Logistic_Regression', 'value': 'Logistic_Regression'}, {'label': 'DecisionTreeClassifier', 'value': 'DecisionTreeClassifier'},
                                 {'label': 'SVM', 'value': 'SVM'}]
            type_algo = "Classification"
        else:
            models_dd_options = [{'label': 'DecisionTreeRegressor', 'value': 'DecisionTreeRegressor'}, {'label': 'K_NearNeighbors_Regressor', 'value': 'K_NearNeighbors_Regressor'},
                                 {'label': 'Linear_Regression', 'value': 'Linear_Regression'}]
            type_algo = "Régression"
        text_type_algo = "Sélectionner le ou les algorithmes de *" + type_algo + "* pour effectuer l'apprentissage du modèle : "
    return models_dd_options,models_dd_options,models_dd_options,text_type_algo


#Store data
@app.callback(Output('data_Stored', 'data'),
              Input('datatable-upload', 'contents'))
def store_the_data(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df_General = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df_General.to_dict('records')


#Store variable_cible
@app.callback(Output('variable_cible_Stored', 'data'),
              Input('id_variable_cible', 'value'))
def store_var_cible(varCible):
    return varCible

#Store variables_explicatives
@app.callback(Output('variables_explis_Stored', 'data'),
              Input('id_var_expli', 'value'))
def store_vars_explic(varsExplis):
    return varsExplis

#Store Modele_ML1
@app.callback(Output('modele_ML_1_Stored', 'data'),
              Input('id_Modele_ML1', 'value'))
def store_modele(modele_ML1):
    return modele_ML1

#Store Modele_ML2
@app.callback(Output('modele_ML_2_Stored', 'data'),
              Input('id_Modele_ML2', 'value'))
def store_modele(modele_ML2):
    return modele_ML2

#Store Modele_ML3
@app.callback(Output('modele_ML_3_Stored', 'data'),
              Input('id_Modele_ML3', 'value'))
def store_modele(modele_ML3):
    return modele_ML3

#Store type_algo
@app.callback(Output('type_algo_Stored', 'data'),
              Input('id_type_algos', 'children'))
def store_type_algo(type_algo):
    return type_algo


def get_model(button_valid, varCible, ModeleML):  # X, y, type_model, nom_model, param_depart
    X_columns = [col for col in list(df.columns) if col != varCible]
    X = df.loc[:, X_columns]
    y = df[varCible]
    nom_model = ModeleML

    #A changer ----------
    type_model = 'classif'
    param_depart = {'C': 1, 'penalty':'l1'}
    #----------------------

    start = time()
    if type_model == 'classif':
        # Initiate the modele
        if nom_model == 'Logistic_Regression':
            modele = LogisticRegression(solver='liblinear', max_iter=5000, random_state=1, C=param_depart['C'],
                                        penalty=param_depart['penalty'])
            params = {'C': [0.5, 1, 2, 3], 'penalty': ['l1', 'l2']}
        elif nom_model == 'DecisionTreeClassifier':
            modele = DecisionTreeClassifier(random_state=1, criterion=param_depart['criterion'],
                                            max_depth=param_depart['max_depth'])
            params = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, 12]}
        elif nom_model == 'SVM':
            modele = svm.SVC(C=param_depart['C'], gamma=param_depart['gamma'])  # select the algorithm
            params = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
        selector = RFECV(DecisionTreeClassifier(random_state=0),
                         step=1,
                         min_features_to_select=2,
                         cv=5)
        selector.fit(X, y)
    elif type_model == 'reg':
        if nom_model == 'DecisionTreeRegressor':
            modele = DecisionTreeRegressor()
            params = {"splitter": ["best", "random"],
                      "max_depth": [1, 5, 11],
                      "min_samples_leaf": [1, 5, 10],
                      "min_weight_fraction_leaf": [0.1, 0.5],
                      "max_features": ["auto", "log2", "sqrt", None],
                      "max_leaf_nodes": [None, 10, 50, 90]}
        elif nom_model == 'K_NearNeighbors_Regressor':
            modele = KNeighborsRegressor()
            params = {'n_neighbors': [2, 3, 4, 5, 6], 'weights': ['uniform', 'distance'],
                      'metric': ['minkowski', 'manhattan', 'euclidean']}
        elif nom_model == "Linear_Regression":
            modele = LinearRegression()
            params = {'fit_intercept': [True, False],
                      'normalize': [True, False], 'copy_X': [True, False]}
        selector = RFECV(DecisionTreeRegressor(random_state=0),
                         step=2,
                         min_features_to_select=2,
                         cv=3)
        selector.fit(X, y)

    print("Avant :", len(X.columns), 'colonnes')
    X = X[np.array(X.columns)[selector.get_support()]]
    print("Apres :", len(X.columns), 'colonnes')
    # Si 1 des parametres est manquant (min) voir les deux
    if not all([param_depart[r] for r in list(param_depart.keys())]):
        clf = GridSearchCV(modele, param_grid=params, cv=5, n_jobs=-1)  # -1 pour paraleliser (+ rapide)
        clf.fit(X, y)
        for i in param_depart.keys():
            if param_depart[i] == None:
                param_depart[i] = clf.best_params_[i]
                print('Parametre', i, 'improved')

        if type_model == 'classif':
            if nom_model == 'Logistic_Regression':
                modele = LogisticRegression(random_state=1, C=param_depart['C'], penalty=param_depart['penalty'])
            elif nom_model == 'DecisionTreeClassifier':
                modele = DecisionTreeClassifier(random_state=1, criterion=param_depart['criterion'],
                                                max_depth=param_depart['max_depth'])
            elif nom_model == 'SVM':
                modele = svm.SVC(C=param_depart['C'], gamma=param_depart['gamma'])  # select the algorithm
        if type_model == 'reg':
            if nom_model == 'DecTreeReg':
                modele = DecisionTreeRegressor(max_depth=param_depart["max_depth"],
                                               max_features=param_depart["max_features"],
                                               max_leaf_nodes=param_depart["max_leaf_nodes"],
                                               min_samples_leaf=param_depart["min_samples_leaf"],
                                               min_weight_fraction_leaf=param_depart["min_weight_fraction_leaf"],
                                               splitter=param_depart["splitter"])
            elif nom_model == 'K_NearNeighbors_Regressor':
                modele = KNeighborsRegressor(n_neighbors=param_depart["n_neighbors"],
                                             weights=param_depart["weights"], metric=param_depart["metric"])
            elif nom_model == "Linear_Regression":
                modele = LinearRegression(fit_intercept=param_depart["fit_intercept"],
                                          normalize=param_depart["normalize"], copy_X=param_depart["copy_X"])

        scores = cross_val_score(modele, X, y, cv=5)  # ,scoring=make_scorer(f1_score(average='micro')) MARCHE PAS???
        print("Moyenne des scores après GridSearch + cross-validation :", np.mean(scores))
    else:
        # Si l'utilisateur entre tous les para
        scores = cross_val_score(modele, X, y, cv=5)  # ,scoring=make_scorer(f1_score) MARCHE PAS???
        print("Moyenne des scores après cross-validation :", np.mean(scores))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)

    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    print('y_pred')
    if type_model == 'classif':
        print('The accuracy is', metrics.accuracy_score(y_pred, y_test))

        # Efficacité
        print(accuracy_score(y_test, y_pred))
        # Matrice de confusion
        print(confusion_matrix(y_test, y_pred))

        # 6. Affichez la précision, le rappel et le F1-score.
        print(precision_score(y_test, y_pred, average='micro'),
              recall_score(y_test, y_pred, average='micro'),
              f1_score(y_test, y_pred, average='micro'))

        # Affichage PCA (prob de couleur + question de si PCA meilleurs visu)
        labels = np.unique(y_test)
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_test)
        pio.renderers.default = 'browser'
        fig1 = px.scatter(components, x=0, y=1, color=y_pred, labels=labels)
        fig2 = px.scatter(components, x=0, y=1, color=y_test, labels=labels)

    elif type_model == 'reg':
        MAE_1 = metrics.mean_absolute_error(y_test, y_pred)
        MSE_1 = metrics.mean_squared_error(y_test, y_pred)
        RMSE_1 = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print("MAE is : ", MAE_1, "\nMSE is : ", MSE_1, "\nRMSE is : ", RMSE_1, "\n")

        plt.scatter(range(len(y_test)), y_pred[np.argsort(y_test)], color="green")  # Prédictions
        plt.plot(range(len(y_test)), np.sort(y_test), color="red")  # Données réelles
        plt.title("Y_pred en vert, y_test en rouge")
        plt.show()

    end = time()
    DeltaTime = end - start
    print("Time spent is : ", DeltaTime)

    return fig1, fig2



# Update the index
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))

def display_page(pathname):
    if pathname == '/page-1': # '/page-1'
        return page_1_layout#, fig1,fig2
        #return_value[1] = {'height': 'auto', 'display': 'inline-block'}
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    else:
        return index_page

@app.callback(
              Output('results_graph_1_PAGE1', 'figure'),
              Output('results_graph_2_PAGE1', 'figure'),
              Output('id_res1_1', 'children'),
              Output('id_res2_1', 'children'),
              Output('id_res3_1', 'children'),
              Output('id_res4_1', 'children'),
              Output('id_res5_1', 'children'),
              Output('id_nom_algo1', 'children'),
              #Output('test_on_submit_button_PAGE3', 'children'), #test_on_submit_button_PAGE1  test_on_submit_button
              Input('submit_val_PAGE1', 'n_clicks'),#submit_val_PAGE1     submit_val
              Input('data_Stored', 'data'),
              Input('variable_cible_Stored', 'data'),
              Input('variables_explis_Stored','data'),
              Input('modele_ML_1_Stored', 'data'),
              Input('type_algo_Stored', 'data'))

def get_model1(button_valid,donnees,varCible,var_explis,ModeleML,type_model):  # X, y, type_model, nom_model, param_depart   var_explis

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'submit_val_PAGE1' in changed_id:
        df_General = pd.DataFrame(donnees)

        if str(var_explis) == "['ALL']":
            X_columns = [col for col in list(df_General.columns) if col != varCible]
            X = df_General.loc[:, X_columns]
            y = df_General[varCible]
        else:
            X = df_General.loc[:, var_explis]
            y = df_General[varCible]  # Classification    Régression

        type_model = str(str(type_model).split("*")[1])


        start = time()

        X = pd.get_dummies(X)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        titre_page = "Résultats pour l'algorithme de " + str(ModeleML)

        # Initialisation des modèles et de leurs paramètres à tester avec GridSearch
        if type_model == 'Classification':
            if ModeleML == 'Logistic_Regression':
                modele = LogisticRegression(solver="liblinear")
                params = [{'penalty': ['l1', 'l2'],
                           'C': np.logspace(-4, 4, 5),
                           'max_iter': [100, 300, 900]}]

            elif ModeleML == 'DecisionTreeClassifier':
                modele = DecisionTreeClassifier()
                params = {'criterion': ['gini', 'entropy'],
                          'max_depth': [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'SVM':
                modele = svm.SVC(probability=True)
                params = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        elif type_model == 'Régression':
            if ModeleML == 'DecisionTreeRegressor':
                modele = DecisionTreeRegressor()
                params = {"splitter": ["best", "random"],
                          "max_depth": [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'K_NearNeighbors_Regressor':
                modele = KNeighborsRegressor()
                params = {'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

            elif ModeleML == "Linear_Regression":
                modele = LinearRegression()
                params = {'fit_intercept': [True, False],
                          'normalize': [True, False],
                          'copy_X': [True, False]}

        # Test sur chaque combinaison de paramètres
        best_model = GridSearchCV(modele, param_grid=params, cv=5, n_jobs=-1)  # -1 pour paraleliser (+ rapide)
        best_model.fit(X, y)

        for i in best_model.best_params_.keys():
            print('Parameter', i, 'set to : ', best_model.best_params_[i])

        # Apprentissage en cross validation
        y_pred = cross_val_predict(best_model, X, y, cv=5)

        # Calcul et affichage des résultats
        if type_model == 'Classification':

            accuracy = accuracy_score(y, y_pred)
            confu_matrix = confusion_matrix(y, y_pred)

            # liens pour comprendre average et ses valeurs
            # https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary
            # https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case

            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')

            text_accuracy = "L'accuracy est de : " + str(accuracy)
            text_recall = "Le ReCall est de : " + str(recall)
            text_precision = "La precision est de " + str(precision)
            text_f1 = "Le F1 est de : " + str(f1)

            print("The accuracy is", accuracy)
            print('The confu_matrix is \n', confu_matrix)
            print('The precision is ', precision)
            print('The recall is ', recall)
            print('The f1 is ', f1)



            # PCA
            labels = np.unique(y)
            pca = PCA(n_components=2)
            components = pd.DataFrame(pca.fit_transform(X), columns=['comp1', 'comp2'])
            components['Prediction'] = y_pred
            components['True value'] = y
            pio.renderers.default = 'browser'
            PCA_pred = px.scatter(components, x='comp1', y='comp2', color='Prediction', labels=labels)
            PCA_true = px.scatter(components, x='comp1', y='comp2', color='True value', labels=labels)

            # courbe ROC de chaque classe
            y_onehot = pd.get_dummies(y, columns=best_model.classes_)
            y_scores = cross_val_predict(best_model, X, y, cv=5, method='predict_proba')
            fig_ROC = go.Figure()
            fig_ROC.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig_ROC.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500)

            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return PCA_pred,PCA_true, text_accuracy, text_precision, text_recall, text_f1,temps_Apprentissage,titre_page

        elif type_model == 'Régression':

            MAE_1 = metrics.mean_absolute_error(y, y_pred)
            MSE_1 = metrics.mean_squared_error(y, y_pred)
            RMSE_1 = np.sqrt(metrics.mean_squared_error(y, y_pred))

            text_MAE1 = "Mean absolute Error est  de : " + str(MAE_1)
            text_MSE_1 = "Mean Square Error est  de : " + str(MSE_1)
            text_RMSE_1 = "Root Mean Square Error est  de : " + str(RMSE_1)
            text_empty = ""  # Adding this text to match with the number of return elemnts

            print("MAE is : ", MAE_1, "\nMSE is : ", MSE_1, "\nRMSE is : ", RMSE_1, "\n")

            # Affichage des résultats
            fig_Scatter = go.Figure(
                data=go.Scatter(name="Prediction", x=list(range(len(y))), y=y_pred[np.argsort(y)], mode='markers'))
            fig_Scatter.add_trace(go.Scatter(name='True value', x=list(range(len(y))), y=np.sort(y),
                                         mode='lines'))
            fig_Fictive = go.Figure()

            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return fig_Scatter, fig_Fictive, text_MAE1, text_MSE_1, text_RMSE_1, text_empty,temps_Apprentissage,titre_page




@app.callback(
              Output('results_graph_1_PAGE2', 'figure'),
              Output('results_graph_2_PAGE2', 'figure'),
              Output('id_res1_2', 'children'),
              Output('id_res2_2', 'children'),
              Output('id_res3_2', 'children'),
              Output('id_res4_2', 'children'),
              Output('id_res5_2', 'children'),
              Output('id_nom_algo2', 'children'),
              #Output('test_on_submit_button_PAGE3', 'children'), #test_on_submit_button_PAGE1  test_on_submit_button
              Input('submit_val_PAGE2', 'n_clicks'),#submit_val_PAGE1     submit_val
              Input('data_Stored', 'data'),
              Input('variable_cible_Stored', 'data'),
              Input('variables_explis_Stored','data'),
              Input('modele_ML_2_Stored', 'data'),
              Input('type_algo_Stored', 'data'))

def get_model2(button_valid,donnees,varCible,var_explis,ModeleML,type_model):  # X, y, type_model, nom_model, param_depart   var_explis

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'submit_val_PAGE2' in changed_id:
        df_General = pd.DataFrame(donnees)

        if str(var_explis) == "['ALL']":
            X_columns = [col for col in list(df_General.columns) if col != varCible]
            X = df_General.loc[:, X_columns]
            y = df_General[varCible]
        else:
            X = df_General.loc[:, var_explis]
            y = df_General[varCible]  # Classification    Régression

        type_model = str(str(type_model).split("*")[1])


        start = time()

        X = pd.get_dummies(X)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        titre_page = "Résultats pour l'algorithme de " + str(ModeleML)

        # Initialisation des modèles et de leurs paramètres à tester avec GridSearch
        if type_model == 'Classification':
            if ModeleML == 'Logistic_Regression':
                modele = LogisticRegression(solver="liblinear")
                params = [{'penalty': ['l1', 'l2'],
                           'C': np.logspace(-4, 4, 5),
                           'max_iter': [100, 300, 900]}]

            elif ModeleML == 'DecisionTreeClassifier':
                modele = DecisionTreeClassifier()
                params = {'criterion': ['gini', 'entropy'],
                          'max_depth': [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'SVM':
                modele = svm.SVC(probability=True)
                params = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        elif type_model == 'Régression':
            if ModeleML == 'DecisionTreeRegressor':
                modele = DecisionTreeRegressor()
                params = {"splitter": ["best", "random"],
                          "max_depth": [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'K_NearNeighbors_Regressor':
                modele = KNeighborsRegressor()
                params = {'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

            elif ModeleML == "Linear_Regression":
                modele = LinearRegression()
                params = {'fit_intercept': [True, False],
                          'normalize': [True, False],
                          'copy_X': [True, False]}

        # Test sur chaque combinaison de paramètres
        best_model = GridSearchCV(modele, param_grid=params, cv=5, n_jobs=-1)  # -1 pour paraleliser (+ rapide)
        best_model.fit(X, y)

        for i in best_model.best_params_.keys():
            print('Parameter', i, 'set to : ', best_model.best_params_[i])

        # Apprentissage en cross validation
        y_pred = cross_val_predict(best_model, X, y, cv=5)

        # Calcul et affichage des résultats
        if type_model == 'Classification':

            accuracy = accuracy_score(y, y_pred)
            confu_matrix = confusion_matrix(y, y_pred)

            # liens pour comprendre average et ses valeurs
            # https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary
            # https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case

            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')

            text_accuracy = "L'accuracy est de : " + str(accuracy)
            text_recall = "Le ReCall est de : " + str(recall)
            text_precision = "La precision est de " + str(precision)
            text_f1 = "Le F1 est de : " + str(f1)

            print("The accuracy is", accuracy)
            print('The confu_matrix is \n', confu_matrix)
            print('The precision is ', precision)
            print('The recall is ', recall)
            print('The f1 is ', f1)

            # PCA
            labels = np.unique(y)
            pca = PCA(n_components=2)
            components = pd.DataFrame(pca.fit_transform(X), columns=['comp1', 'comp2'])
            components['Prediction'] = y_pred
            components['True value'] = y
            pio.renderers.default = 'browser'
            PCA_pred = px.scatter(components, x='comp1', y='comp2', color='Prediction', labels=labels)
            PCA_true = px.scatter(components, x='comp1', y='comp2', color='True value', labels=labels)

            # courbe ROC de chaque classe
            y_onehot = pd.get_dummies(y, columns=best_model.classes_)
            y_scores = cross_val_predict(best_model, X, y, cv=5, method='predict_proba')
            fig_ROC = go.Figure()
            fig_ROC.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig_ROC.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500)

            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return PCA_pred,PCA_true, text_accuracy, text_precision, text_recall, text_f1,temps_Apprentissage,titre_page

        elif type_model == 'Régression':

            MAE_1 = metrics.mean_absolute_error(y, y_pred)
            MSE_1 = metrics.mean_squared_error(y, y_pred)
            RMSE_1 = np.sqrt(metrics.mean_squared_error(y, y_pred))

            text_MAE1 = "Mean absolute Error est  de : " + str(MAE_1)
            text_MSE_1 = "Mean Square Error est  de : " + str(MSE_1)
            text_RMSE_1 = "Root Mean Square Error est  de : " + str(RMSE_1)
            text_empty = ""  # Adding this text to match with the number of return elemnts

            print("MAE is : ", MAE_1, "\nMSE is : ", MSE_1, "\nRMSE is : ", RMSE_1, "\n")

            # Affichage des résultats
            fig_Scatter = go.Figure(
                data=go.Scatter(name="Prediction", x=list(range(len(y))), y=y_pred[np.argsort(y)], mode='markers'))
            fig_Scatter.add_trace(go.Scatter(name='True value', x=list(range(len(y))), y=np.sort(y),
                                         mode='lines'))
            fig_Fictive = go.Figure()

            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return fig_Scatter, fig_Fictive, text_MAE1, text_MSE_1, text_RMSE_1, text_empty,temps_Apprentissage,titre_page


'''
def get_model2(button_valid,donnees,varCible,var_explis,ModeleML,type_model):  # X, y, type_model, nom_model, param_depart   var_explis

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'submit_val_PAGE2' in changed_id:

        #textt = "voilaaaa " + var_explis + " --- " + ModeleML
        df_General = pd.DataFrame(donnees)
        #X_columns = [col for col in list(df_General.columns) if col != varCible]
        #X = df_General.loc[:, X_columns]
        #y = df_General[varCible]

        X = df_General.loc[:,var_explis]
        y = df_General[varCible]     # Classification    Régression

        #nom_model = ModeleML

        type_model = str(str(type_model).split("*")[1])
        #A changer ----------
        #type_model = 'classif'
        #param_depart = {'solver':None, 'C': None, 'penalty':None}  #test Logistic_Regression
        #param_depart = {'criterion': 'gini', 'max_depth':10}  #test DecisionTreeClassifier
        #param_depart = {'C': 1, 'gamma':0.01}  #test SVM
        #param_depart = {'n_neighbors':None, 'weights': None, 'algorithm':None}  # test K_NearNeighbors_Regressor
        #----------------------

        start = time()

        X = pd.get_dummies(X)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        titre_page = "Résultats pour l'algorithme de " + str(ModeleML)

        # Initialisation des modèles et de leurs paramètres à tester avec GridSearch
        if type_model == 'Classification':
            if ModeleML == 'Logistic_Regression':
                modele = LogisticRegression(solver="liblinear")
                # Some penalties does not work with some solvers
                params = [{'penalty': ['l1', 'l2'],
                           'C': np.logspace(-4, 4, 5),
                           'max_iter': [100, 300, 900]}]

            elif ModeleML == 'DecisionTreeClassifier':
                modele = DecisionTreeClassifier()
                params = {'criterion': ['gini', 'entropy'],
                          'max_depth': [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'SVM':
                modele = svm.SVC(probability=True)
                params = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        elif type_model == 'Régression':
            if ModeleML == 'DecisionTreeRegressor':
                modele = DecisionTreeRegressor()
                params = {"splitter": ["best", "random"],
                          "max_depth": [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'K_NearNeighbors_Regressor':
                modele = KNeighborsRegressor()
                params = {'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

            elif ModeleML == "Linear_Regression":
                modele = LinearRegression()
                params = {'fit_intercept': [True, False],
                          'normalize': [True, False],
                          'copy_X': [True, False]}

        # Test sur chaque combinaison de paramètres
        best_model = GridSearchCV(modele, param_grid=params, cv=5, n_jobs=-1)  # -1 pour paraleliser (+ rapide)
        best_model.fit(X, y)

        for i in best_model.best_params_.keys():
            print('Parameter', i, 'set to : ', best_model.best_params_[i])

        # Apprentissage en cross validation
        y_pred = cross_val_predict(best_model, X, y, cv=5)

        # Calcul et affichage des résultats
        if type_model == 'Classification':

            accuracy = accuracy_score(y, y_pred)
            confu_matrix = confusion_matrix(y, y_pred)

            # liens pour comprendre average et ses valeurs
            # https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary
            # https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case

            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')

            text_accuracy = "L'accuracy est de : " + str(accuracy)
            text_recall = "Le ReCall est de : " + str(recall)
            text_precision = "La precision est de " + str(precision)
            text_f1 = "Le F1 est de : " + str(f1)

            print("The accuracy is", accuracy)
            print('The confu_matrix is \n', confu_matrix)
            print('The precision is ', precision)
            print('The recall is ', recall)
            print('The f1 is ', f1)



            # PCA
            labels = np.unique(y)
            pca = PCA(n_components=2)
            components = pd.DataFrame(pca.fit_transform(X), columns=['comp1', 'comp2'])
            components['Prediction'] = y_pred
            components['True value'] = y
            pio.renderers.default = 'browser'
            PCA_pred = px.scatter(components, x='comp1', y='comp2', color='Prediction', labels=labels)
            PCA_true = px.scatter(components, x='comp1', y='comp2', color='True value', labels=labels)

            # PCA_pred.show()
            # PCA_true.show()

            # courbe ROC de chaque classe
            y_onehot = pd.get_dummies(y, columns=best_model.classes_)
            y_scores = cross_val_predict(best_model, X, y, cv=5, method='predict_proba')
            fig_ROC = go.Figure()
            fig_ROC.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig_ROC.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500)
            # fig_ROC.show()
            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return PCA_pred,PCA_true, text_accuracy, text_precision, text_recall, text_f1,temps_Apprentissage,titre_page

        elif type_model == 'Régression':

            MAE_1 = metrics.mean_absolute_error(y, y_pred)
            MSE_1 = metrics.mean_squared_error(y, y_pred)
            RMSE_1 = np.sqrt(metrics.mean_squared_error(y, y_pred))

            text_MAE1 = "Mean absolute Error est  de : " + str(MAE_1)
            text_MSE_1 = "Mean Square Error est  de : " + str(MSE_1)
            text_RMSE_1 = "Root Mean Square Error est  de : " + str(RMSE_1)
            text_empty = ""  # Adding this text to match with the number of return elemnts

            print("MAE is : ", MAE_1, "\nMSE is : ", MSE_1, "\nRMSE is : ", RMSE_1, "\n")

            # Affichage des résultats
            fig_Scatter = go.Figure(
                data=go.Scatter(name="Prediction", x=list(range(len(y))), y=y_pred[np.argsort(y)], mode='markers'))
            fig_Scatter.add_trace(go.Scatter(name='True value', x=list(range(len(y))), y=np.sort(y),
                                         mode='lines'))
            fig_Fictive = go.Figure()
            # fig_Scatter.show()
            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return fig_Scatter, fig_Fictive, text_MAE1, text_MSE_1, text_RMSE_1, text_empty,temps_Apprentissage,titre_page
'''
@app.callback(
              Output('results_graph_1_PAGE3', 'figure'),
              Output('results_graph_2_PAGE3', 'figure'),
              #Output('graph_container', 'style'),
              Output('id_res1_3', 'children'),
              Output('id_res2_3', 'children'),
              Output('id_res3_3', 'children'),
              Output('id_res4_3', 'children'),
              Output('id_res5_3', 'children'),
              Output('id_nom_algo3', 'children'),
              #Output('test_on_submit_button_PAGE3', 'children'), #test_on_submit_button_PAGE1  test_on_submit_button
              Input('submit_val_PAGE3', 'n_clicks'),
              Input('data_Stored', 'data'),
              Input('variable_cible_Stored', 'data'),
              Input('variables_explis_Stored','data'),
              Input('modele_ML_3_Stored', 'data'),
              Input('type_algo_Stored', 'data'))

def get_model3(button_valid,donnees,varCible,var_explis,ModeleML,type_model):  # X, y, type_model, nom_model, param_depart   var_explis

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'submit_val_PAGE3' in changed_id:
        df_General = pd.DataFrame(donnees)

        if str(var_explis) == "['ALL']":
            X_columns = [col for col in list(df_General.columns) if col != varCible]
            X = df_General.loc[:, X_columns]
            y = df_General[varCible]
        else:
            X = df_General.loc[:, var_explis]
            y = df_General[varCible]  # Classification    Régression

        type_model = str(str(type_model).split("*")[1])


        start = time()

        X = pd.get_dummies(X)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        titre_page = "Résultats pour l'algorithme de " + str(ModeleML)

        # Initialisation des modèles et de leurs paramètres à tester avec GridSearch
        if type_model == 'Classification':
            if ModeleML == 'Logistic_Regression':
                modele = LogisticRegression(solver="liblinear")
                # Some penalties does not work with some solvers
                params = [{'penalty': ['l1', 'l2'],
                           'C': np.logspace(-4, 4, 5),
                           'max_iter': [100, 300, 900]}]

            elif ModeleML == 'DecisionTreeClassifier':
                modele = DecisionTreeClassifier()
                params = {'criterion': ['gini', 'entropy'],
                          'max_depth': [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'SVM':
                modele = svm.SVC(probability=True)
                params = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        elif type_model == 'Régression':
            if ModeleML == 'DecisionTreeRegressor':
                modele = DecisionTreeRegressor()
                params = {"splitter": ["best", "random"],
                          "max_depth": [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'K_NearNeighbors_Regressor':
                modele = KNeighborsRegressor()
                params = {'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

            elif ModeleML == "Linear_Regression":
                modele = LinearRegression()
                params = {'fit_intercept': [True, False],
                          'normalize': [True, False],
                          'copy_X': [True, False]}

        # Test sur chaque combinaison de paramètres
        best_model = GridSearchCV(modele, param_grid=params, cv=5, n_jobs=-1)  # -1 pour paraleliser (+ rapide)
        best_model.fit(X, y)

        for i in best_model.best_params_.keys():
            print('Parameter', i, 'set to : ', best_model.best_params_[i])

        # Apprentissage en cross validation
        y_pred = cross_val_predict(best_model, X, y, cv=5)

        # Calcul et affichage des résultats
        if type_model == 'Classification':

            accuracy = accuracy_score(y, y_pred)
            confu_matrix = confusion_matrix(y, y_pred)

            # liens pour comprendre average et ses valeurs
            # https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary
            # https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case

            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')

            text_accuracy = "L'accuracy est de : " + str(accuracy)
            text_recall = "Le ReCall est de : " + str(recall)
            text_precision = "La precision est de " + str(precision)
            text_f1 = "Le F1 est de : " + str(f1)

            print("The accuracy is", accuracy)
            print('The confu_matrix is \n', confu_matrix)
            print('The precision is ', precision)
            print('The recall is ', recall)
            print('The f1 is ', f1)

            # PCA
            labels = np.unique(y)
            pca = PCA(n_components=2)
            components = pd.DataFrame(pca.fit_transform(X), columns=['comp1', 'comp2'])
            components['Prediction'] = y_pred
            components['True value'] = y
            pio.renderers.default = 'browser'
            PCA_pred = px.scatter(components, x='comp1', y='comp2', color='Prediction', labels=labels)
            PCA_true = px.scatter(components, x='comp1', y='comp2', color='True value', labels=labels)

            # courbe ROC de chaque classe
            y_onehot = pd.get_dummies(y, columns=best_model.classes_)
            y_scores = cross_val_predict(best_model, X, y, cv=5, method='predict_proba')
            fig_ROC = go.Figure()
            fig_ROC.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig_ROC.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500)

            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return PCA_pred,PCA_true, text_accuracy, text_precision, text_recall, text_f1,temps_Apprentissage,titre_page

        elif type_model == 'Régression':

            MAE_1 = metrics.mean_absolute_error(y, y_pred)
            MSE_1 = metrics.mean_squared_error(y, y_pred)
            RMSE_1 = np.sqrt(metrics.mean_squared_error(y, y_pred))

            text_MAE1 = "Mean absolute Error est  de : " + str(MAE_1)
            text_MSE_1 = "Mean Square Error est  de : " + str(MSE_1)
            text_RMSE_1 = "Root Mean Square Error est  de : " + str(RMSE_1)
            text_empty = ""  # Adding this text to match with the number of return elemnts

            print("MAE is : ", MAE_1, "\nMSE is : ", MSE_1, "\nRMSE is : ", RMSE_1, "\n")

            # Affichage des résultats
            fig_Scatter = go.Figure(
                data=go.Scatter(name="Prediction", x=list(range(len(y))), y=y_pred[np.argsort(y)], mode='markers'))
            fig_Scatter.add_trace(go.Scatter(name='True value', x=list(range(len(y))), y=np.sort(y),
                                         mode='lines'))
            fig_Fictive = go.Figure()
            # fig_Scatter.show()
            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return fig_Scatter, fig_Fictive, text_MAE1, text_MSE_1, text_RMSE_1, text_empty,temps_Apprentissage,titre_page



'''
def get_model3(button_valid,donnees,varCible,var_explis,ModeleML,type_model):  # X, y, type_model, nom_model, param_depart   var_explis

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'submit_val_PAGE3' in changed_id:

        #textt = "voilaaaa " + var_explis + " --- " + ModeleML
        df_General = pd.DataFrame(donnees)
        #X_columns = [col for col in list(df_General.columns) if col != varCible]
        #X = df_General.loc[:, X_columns]
        #y = df_General[varCible]

        X = df_General.loc[:,var_explis]
        y = df_General[varCible]     # Classification    Régression

        #nom_model = ModeleML

        type_model = str(str(type_model).split("*")[1])
        #A changer ----------
        #type_model = 'classif'
        #param_depart = {'solver':None, 'C': None, 'penalty':None}  #test Logistic_Regression
        #param_depart = {'criterion': 'gini', 'max_depth':10}  #test DecisionTreeClassifier
        #param_depart = {'C': 1, 'gamma':0.01}  #test SVM
        #param_depart = {'n_neighbors':None, 'weights': None, 'algorithm':None}  # test K_NearNeighbors_Regressor
        #----------------------

        start = time()

        X = pd.get_dummies(X)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

        titre_page = "Résultats pour l'algorithme de " + str(ModeleML)
        # Initialisation des modèles et de leurs paramètres à tester avec GridSearch
        if type_model == 'Classification':
            if ModeleML == 'Logistic_Regression':
                modele = LogisticRegression(solver="liblinear")
                # Some penalties does not work with some solvers
                params = [{'penalty': ['l1', 'l2'],
                           'C': np.logspace(-4, 4, 5),
                           'max_iter': [100, 300, 900]}]

            elif ModeleML == 'DecisionTreeClassifier':
                modele = DecisionTreeClassifier()
                params = {'criterion': ['gini', 'entropy'],
                          'max_depth': [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'SVM':
                modele = svm.SVC(probability=True)
                params = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

        elif type_model == 'Régression':
            if ModeleML == 'DecisionTreeRegressor':
                modele = DecisionTreeRegressor()
                params = {"splitter": ["best", "random"],
                          "max_depth": [None, 2, 5, 10, 15],
                          'max_features': ['auto', 'sqrt', 'log2']}

            elif ModeleML == 'K_NearNeighbors_Regressor':
                modele = KNeighborsRegressor()
                params = {'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                          'weights': ['uniform', 'distance'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

            elif ModeleML == "Linear_Regression":
                modele = LinearRegression()
                params = {'fit_intercept': [True, False],
                          'normalize': [True, False],
                          'copy_X': [True, False]}

        # Test sur chaque combinaison de paramètres
        best_model = GridSearchCV(modele, param_grid=params, cv=5, n_jobs=-1)  # -1 pour paraleliser (+ rapide)
        best_model.fit(X, y)

        for i in best_model.best_params_.keys():
            print('Parameter', i, 'set to : ', best_model.best_params_[i])

        # Apprentissage en cross validation
        y_pred = cross_val_predict(best_model, X, y, cv=5)

        # Calcul et affichage des résultats
        if type_model == 'Classification':

            accuracy = accuracy_score(y, y_pred)
            confu_matrix = confusion_matrix(y, y_pred)

            # liens pour comprendre average et ses valeurs
            # https://stackoverflow.com/questions/52269187/facing-valueerror-target-is-multiclass-but-average-binary
            # https://stackoverflow.com/questions/31421413/how-to-compute-precision-recall-accuracy-and-f1-score-for-the-multiclass-case

            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')

            text_accuracy = "L'accuracy est de : " + str(accuracy)
            text_recall = "Le ReCall est de : " + str(recall)
            text_precision = "La precision est de " + str(precision)
            text_f1 = "Le F1 est de : " + str(f1)

            print("The accuracy is", accuracy)
            print('The confu_matrix is \n', confu_matrix)
            print('The precision is ', precision)
            print('The recall is ', recall)
            print('The f1 is ', f1)



            # PCA
            labels = np.unique(y)
            pca = PCA(n_components=2)
            components = pd.DataFrame(pca.fit_transform(X), columns=['comp1', 'comp2'])
            components['Prediction'] = y_pred
            components['True value'] = y
            pio.renderers.default = 'browser'
            PCA_pred = px.scatter(components, x='comp1', y='comp2', color='Prediction', labels=labels)
            PCA_true = px.scatter(components, x='comp1', y='comp2', color='True value', labels=labels)

            # PCA_pred.show()
            # PCA_true.show()

            # courbe ROC de chaque classe
            y_onehot = pd.get_dummies(y, columns=best_model.classes_)
            y_scores = cross_val_predict(best_model, X, y, cv=5, method='predict_proba')
            fig_ROC = go.Figure()
            fig_ROC.add_shape(
                type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1
            )
            for i in range(y_scores.shape[1]):
                y_true = y_onehot.iloc[:, i]
                y_score = y_scores[:, i]

                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)

                name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
                fig_ROC.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

            fig_ROC.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=700, height=500)
            # fig_ROC.show()

            new_style = {'display': 'block'}

            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return PCA_pred,PCA_true, text_accuracy, text_precision, text_recall, text_f1,temps_Apprentissage,titre_page

        elif type_model == 'Régression':

            MAE_1 = metrics.mean_absolute_error(y, y_pred)
            MSE_1 = metrics.mean_squared_error(y, y_pred)
            RMSE_1 = np.sqrt(metrics.mean_squared_error(y, y_pred))

            text_MAE1 = "Mean absolute Error est  de : " + str(MAE_1)
            text_MSE_1 = "Mean Square Error est  de : " + str(MSE_1)
            text_RMSE_1 = "Root Mean Square Error est  de : " + str(RMSE_1)
            text_empty = ""  # Adding this text to match with the number of return elemnts

            print("MAE is : ", MAE_1, "\nMSE is : ", MSE_1, "\nRMSE is : ", RMSE_1, "\n")

            # Affichage des résultats
            fig_Scatter = go.Figure(
                data=go.Scatter(name="Prediction", x=list(range(len(y))), y=y_pred[np.argsort(y)], mode='markers'))
            fig_Scatter.add_trace(go.Scatter(name='True value', x=list(range(len(y))), y=np.sort(y),
                                         mode='lines'))
            fig_Fictive = go.Figure()
            new_style = {'display':'none'}
            # fig_Scatter.show()
            end = time()
            DeltaTime = end - start
            temps_Apprentissage = "Temps d'apprentissage du modèle : " + str(DeltaTime) + " Secondes"

            return fig_Scatter, fig_Fictive, text_MAE1, text_MSE_1, text_RMSE_1, text_empty,temps_Apprentissage,titre_page

'''



if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1')