import streamlit as st 
import pandas as pd
import seaborn as sns
import requests
import shap
from shap.plots import waterfall
import matplotlib.pyplot as plt

#Configurations
st.set_page_config( 
    page_title="Probabilité de faillite d'un client",
    page_icon="loan.png" 
)
st.set_option('deprecation.showPyplotGlobalUse', False)

##############################
# Fonctions
##############################

# Shap summary global
#@st.cache
#def shap_summary(shap_vals,features):
#    return shap.summary_plot(shap_vals, feature_names=features)

# Graphique shap local
@st.cache
def waterfall_plot(nb, ft, expected_val, shap_val):
    return shap.plots._waterfall.waterfall_legacy(expected_val, shap_val, max_display=nb, feature_names=ft)

@st.cache
def afficherScore(score):
    # Affichage du score:
    st.markdown('**La probabilité de faillite de rembourser du prêt par le client est de** ' + str(score))
    # Probabilité par rapport au seuil fixé
    if score < 0.5: 
        st.error('Attention! Le demandeur a une grande probabilté de ne pas rembourser le prêt!') 
    else: 
        st.success("C'est bon! Le demandeur a une grande probabilité de rembourser son prêt!")

######################
# main page layout
######################

st.title("Probabilité de faillite d'un client")
st.subheader("Cette application de Machine learning vous permettra de prédire si un demandeur de prêt sera en mesure de le rembourser ou non.")

col1, col2 = st.columns([1, 1])

with col1:
    st.image("loan.png")

with col2:
    st.write("""L’entreprise, Prêt à dépenser, souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, 
puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données 
variées (données comportementales, données provenant d'autres institutions financières, etc.).

Prêt à dépenser a donc décidé de développer ce dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus 
transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les 
explorer facilement. """)

st.subheader("Pour prédire le statut de défaut/échec de remboursement, vous devez suivre les étapes ci-dessous:")
st.markdown("""
1. Entrez le numéro identifiant de la personne faisant une demande de prêt ;
2. Appuyez sur le bouton "Prédire" et attendez le résultat.

""")

st.subheader("Vous pouvez trouver ci-dessous la probabilité de remboursement du prêt par le demandeur:")

######################
# sidebar layout
######################

st.sidebar.title("Numéro Identifiant du demandeur de prêt")
st.sidebar.image("ab.png", width=100)
st.sidebar.write("Entrer ici le numéro indentifiant du demandeur de prêt dans la base de données:")

liste = pd.read_csv('liste_sampled.csv',delimiter= ',')
option1 = st.sidebar.selectbox(
    'Quel est le numéro identifiant du demandeur?',
    liste)
st.sidebar.write('Vous avez choisi:', option1)

graphique = pd.read_csv('graphique.csv',delimiter= ',')
graphique.set_index('SK_ID_CURR',inplace=True)
option2 = st.sidebar.selectbox(
    'Quelle première variable souhaitez-vous analyser par rapport au demandeur?',
    graphique.columns)
st.sidebar.write('Vous avez choisi:', option2)

option3 = st.sidebar.selectbox(
    'Quelle seconde variable souhaitez-vous analyser par rapport au demandeur?',
    graphique.columns)
st.sidebar.write('Vous avez choisi:', option3)

#Liste des features quantitatives ou discrètes:
quali=["CNT_FAM_MEMBERS","REGION_RATING_CLIENT","LIVE_REGION_NOT_WORK_REGION","OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE","FLAG_DOCUMENT_9","FLAG_DOCUMENT_17"]

##############################
# Connexion avec l'api flask
###############################

btn_predict = st.sidebar.button("Predict")
if btn_predict: 
    response = requests.get("https://stark-lake-17991.herokuapp.com/predict?numeroClient=" + str(option1))
    d = response.json()
    proba=d[0]

    # Affichage du score:
    st.markdown('**La probabilité de faillite de rembourser du prêt par le client est de** ' + str(round(proba, 2)))
    # Probabilité par rapport au seuil fixé
    if proba < 0.5: 
        st.error('Attention! Le demandeur a une grande probabilté de ne pas rembourser le prêt!') 
    else: 
        st.success("C'est bon! Le demandeur a une grande probabilité de rembourser son prêt!")
    
    # Graphiques
    X_train = pd.read_csv('colonnes.csv',delimiter= ',')
    st.markdown('SHAP waterfall Plot pour le client demandeur de prêt:')
    waterfall_plot(10, X_train.columns, d[2], pd.DataFrame.from_dict(d[1]["0"],orient='index').values[:,0])    
    plt.gcf()
    st.pyplot(plt.gcf())

    #st.markdown('SHAP Summary Plot pour le client demandeur de prêt:')
    #shap_summary(json.dumps(d[3]),X_train.columns)    
    #plt.gcf()
    #st.pyplot(plt.gcf())

    if option2 in quali:
        graphique[option2] = graphique[option2].astype(int)
        st.markdown('**Diagramme en barres de** ' + str(option2) + " **:**")
        fig = plt.figure()
        palette=['grey' if (x != int(graphique[option2].loc[[option1]].values)) else 'red' for x in graphique[option2].value_counts().index.sort_values()]
        sns.countplot(x=graphique[option2], palette=palette)
        st.pyplot(fig.figure)
    else:
        st.markdown('**Kdeplot de** ' + str(option2)+ " **:**")
        fig = plt.figure()
        fig = sns.kdeplot(data=graphique, x=option2)
        plt.axvline(int(graphique[option2].loc[[option1]].values),color='r')
        st.pyplot(fig.figure)
    st.write('Voici la valeur du demandeur de prêt par rapport aux données', int(graphique[option2].loc[[option1]].values))

    if option3 in quali:
        graphique[option3] = graphique[option3].astype(int)
        st.markdown('**Diagramme en barres de** ' + str(option3)+ " **:**")
        fig = plt.figure()
        palette=['grey' if (x != int(graphique[option3].loc[[option1]].values)) else 'red' for x in graphique[option3].value_counts().index.sort_values()]
        sns.countplot(x=graphique[option3], palette=palette)
        st.pyplot(fig.figure)
    else:
        st.markdown('**Kdeplot de** ' + str(option3)+ " **:**")
        fig = plt.figure()
        fig = sns.kdeplot(data=graphique, x=option3)
        plt.axvline(int(graphique[option3].loc[[option1]].values),color='r')
        st.pyplot(fig.figure)
    st.write('Voici la valeur du demandeur de prêt par rapport aux données', int(graphique[option3].loc[[option1]].values))
    


