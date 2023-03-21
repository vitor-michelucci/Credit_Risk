#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:49:23 2023

@author: vitormichelucci

Fonte: https://www.kaggle.com/datasets/laotse/credit-risk-dataset
"""

#%% Libraries

import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from scipy import stats # estatística chi2
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import plotly.graph_objs as go # gráfico 3D
import statsmodels.formula.api as smf # estimação do modelo logístico binário
from statstests.process import stepwise # stepwise
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score # matriz de confusão
from sklearn.metrics import roc_curve, auc # curva roc

import warnings
warnings.filterwarnings('ignore')

#%%  Carregando os dados

df = pd.read_csv("credit_risk.csv", encoding = "utf-8")

#%% Visualizando os dataset

df.head()
df.info()
df[['person_age', 'person_emp_length', 'cb_person_cred_hist_length']].describe()

#%% Verificando valores faltantes

df.isna().sum()

#%% Verificando valores duplicados

df.duplicated().sum()

#%% Verificando a cardinalidade

df.nunique()

#%% Removendo valores faltantes

df.dropna(inplace=True)

#%% Removendo valores duplicados

df.drop_duplicates(inplace=True)

#%% Renomeando variáveis

df = df.rename(columns={'person_age':'idade', 'person_income':'renda', 'person_home_ownership':'tipo_moradia', 
                        'person_emp_length':'tempo_emprego', 'loan_intent':'finalidade_emp',
                        'loan_grade':'score_emp', 'loan_amnt':'valor_emp', 'loan_int_rate':'tx_jr_emp', 
                        'loan_percent_income':'perc_comp_renda', 'cb_person_default_on_file':'hist_inad',
                        'cb_person_cred_hist_length':'hist_credito', 'loan_status':'inad'})
df.info()

#%% Ordenando as colunas

df = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 8]]

df.info()
df.shape

#%% Renomeando as variáveis categóricas

df['tipo_moradia'].value_counts()
df['finalidade_emp'].value_counts()
df['hist_inad'].value_counts()

df['tipo_moradia'] = df['tipo_moradia'].replace({'RENT': 'ALUGADA', 
                                                 'MORTGAGE': 'FINANCIADA', 
                                                 'OWN': 'PROPRIA',
                                                 'OTHER': 'OUTROS'})

df['finalidade_emp'] = df['finalidade_emp'].replace({'EDUCATION': 'EDUCACAO', 
                                                 'MEDICAL': 'SAUDE', 
                                                 'VENTURE': 'LAZER',
                                                 'PERSONAL': 'PESSOAL',
                                                 'DEBTCONSOLIDATION': 'RENEGOCIACAO',
                                                 'HOMEIMPROVEMENT': 'REFORMA'})

df['hist_inad'] = df['hist_inad'].replace({'Y': 'S'})
                                                

#%% Salvando a base de dados 

df.to_csv('df.credit.risk.csv',index=False)

#%% Modelagem dos dados com regressão logística

#%% Carregando o dataset

df = pd.read_csv("df.credit.risk.csv", encoding = "utf-8")

df.info()

#%% Preparação do dataset

#%%# Fazendo a contagem da variável target

# Contagem absoluta
abs_count = df['inad'].value_counts()

# Contagem relativa
rel_count = df['inad'].value_counts(normalize=True)

# Concatenar as duas séries de dados
result = pd.concat([rel_count, abs_count], axis=1)

# Renomear as colunas
result.columns = ['Proporção', 'Contagem']

# Ordenar os valores pela contagem absoluta
result = result.sort_values(by='Contagem', ascending=False)

# Imprimir o resultado
print(result)

# Transformação das variáveis categóricas não binárias em dummies

# Verificando a quantidade de categorias das variáveis 
df['tipo_moradia'].value_counts()
df['finalidade_emp'].value_counts()
df['score_emp'].value_counts()
df['hist_inad'].value_counts()

# get dummies
df_dummies = pd.get_dummies(df,
                            columns=['tipo_moradia',
                                     'finalidade_emp',
                                     'score_emp',
                                     'hist_inad'],
                                      drop_first=True)

df_dummies.info()

#%% Estimação do modelo logístico binário

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_dummies.drop(columns=['inad']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "inad ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

#Modelo propriamente dito
modelo_credito = sm.Logit.from_formula(formula_dummies_modelo, 
                                       df_dummies).fit()

#Parâmetros do modelo
modelo_credito.summary()

#%% Procedimento Stepwise

#Estimação do modelo por meio do procedimento Stepwise
step_modelo_credito = stepwise(modelo_credito, pvalue_limit=0.05)

# In[ ]: Comparando os parâmetros dos modelos

summary_col([modelo_credito, step_modelo_credito],
            model_names=["MODELO INICIAL","MODELO STEPWISE"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.2f}".format(x.llf)
        })


# In[ ]: Construção de função para a definição da matriz de confusão

def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores


# In[ ]: Construção da matriz de confusão

# Adicionando os valores previstos de probabilidade na base de dados
df_dummies['phat'] = step_modelo_credito.predict()

#Matriz de confusão para cutoff = 0.5
matriz_confusao(observado=df_dummies['inad'],
                predicts=df_dummies['phat'],
                cutoff=0.50)

# In[ ]: Construção da curva ROC

#Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_dummies['inad'],
                                df_dummies['phat'])
roc_auc = auc(fpr, tpr)

#Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

#Plotando a curva ROC
plt.figure(figsize=(10,10))
plt.plot(fpr,tpr, '-o', color='red')
plt.plot(fpr,fpr, ':', color='gray')
plt.title('Área abaixo da curva: %g' % round(roc_auc,4) +
          ' | Coeficiente de GINI: %g' % round(gini,4), fontsize=17)
plt.xlabel('1 - Especificidade', fontsize=15)
plt.ylabel('Sensitividade', fontsize=15)
plt.show()

# In[ ]: Igualando critérios de especificidade e de sensitividade

#Tentaremos estabelecer um critério que iguale a probabilidade de
#acerto daqueles que chegarão atrasados (sensitividade) e a probabilidade de
#acerto daqueles que não chegarão atrasados (especificidade).

#ATENÇÃO: o que será feito a seguir possui fins didáticos, apenas. DE NENHUMA
#FORMA o procedimento garante a maximização da acurácia do modelo!

#Criação da função 'espec_sens' para a construção de um dataset com diferentes
#valores de cutoff, sensitividade e especificidade:

def espec_sens(observado,predicts):
    
    # adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado


# In[ ]: Até o momento, foram extraídos 3 vetores: 'sensitividade',
#'especificidade' e 'cutoffs'. Assim, criamos um dataframe que contém
#os vetores mencionados

dados_plotagem = espec_sens(observado = df_dummies['inad'],
                            predicts = df_dummies['phat'])
dados_plotagem


# In[ ]: Visualizando o novo dataframe 'dados_plotagem' e plotando os dados
#em um gráfico que mostra a variação da especificidade e da sensitividade
#em função do cutoff

plt.figure(figsize=(10,10))
plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, '-o',
         color='indigo')
plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, '-o',
         color='limegreen')
plt.legend(['Sensitividade', 'Especificidade'], fontsize=17)
plt.xlabel('Cuttoff', fontsize=14)
plt.ylabel('Sensitividade / Especificidade', fontsize=14)
plt.show()

#%%
############################   FIM  ###########################################