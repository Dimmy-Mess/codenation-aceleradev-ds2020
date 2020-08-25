#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[17]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import KBinsDiscretizer,OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


#countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

#countries.columns = new_column_names

#countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# Podemos resolver o problema das vírgulas nas variáveis numéricas usando o parâmetro `decimal` no comando `read_csv` do Pandas. Dado o tamanho curto do dataset, é relativamente mais fácil fazer isso que aplicar rotinas e alterar os tipos das variáveis; depois, usar o comando `apply` para remover os espaços nos extremos das variáveis de texto! Para evitar a repetição de instruções, é interessante comentar acima os comandos relacionados à leitura e uso do dataset.

# In[5]:


countries = pd.read_csv('countries.csv',decimal=',')
countries.columns = new_column_names

countries['Country'] = countries['Country'].apply(lambda x: x.strip())
countries['Region'] = countries['Region'].apply(lambda x: x.strip())

countries.head(5)


# In[6]:


# Sua análise começa aqui
aux ={
    'colunas': countries.columns,
    'tipo': countries.dtypes,
    'NA#': countries.isna().sum(),
    'NA%': countries.isna().sum()/countries.shape[0]
}

non_numeric, numeric = [],[]
for i in aux['colunas']:
    if countries[i].dtype == 'object':
        non_numeric.append(i)
    else:
        numeric.append(i)
        
df_aux = pd.DataFrame(aux)
df_aux


# In[7]:


countries[numeric].describe()


# In[8]:


sns.scatterplot(y = np.log(countries['Infant_mortality']), x=np.log(countries['GDP']),hue='Region', data=countries)


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[9]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return sorted(list(countries['Region'].unique()))


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[10]:


def q2():
    # Retorne aqui o resultado da questão 2.
    data = np.array(countries['Pop_density']).reshape(-1,1)
    kbins = KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
    kbins.fit(data)
    disc_data = kbins.transform(data)
    s = kbins.bin_edges_[0]
    return int(sum(disc_data[:,0] >= 9))


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[11]:


def q3():
    encoder = OneHotEncoder(sparse=False,dtype=np.int,handle_unknown='ignore')
    region_encoder = encoder.fit_transform(countries[['Region']]).shape[1]
    climate_encoder = encoder.fit_transform(countries[['Climate']].dropna()).shape[1]
    
    return region_encoder+climate_encoder+1 #+1 por causa do NaN


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[32]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[93]:


def q4():
    # Retorne aqui o resultado da questão 4.
    data = countries.copy()
    cols = data.columns[2:len(data.columns)]
    
    for col in numeric:
        data[col].fillna(data[col].median(),inplace=True)
    
    std = StandardScaler()
    data[cols]=std.fit_transform(data[cols])
    
    sample = np.array(test_country[2:]).reshape(1,-1)
    return float(std.transform(sample)[0][9].round(3))
    
#Abaixo, código que fiz usando o comando sklearn.pipeline.Pipeline. Não entendi o que houve de errado; 
#ele apenas retorna o valor diferente do esperado(0.202). Alguém sabe explicar pq? =D Obrigado!


# def q4():
#     data = countries.copy()[numeric]
#     pipeline = sk.pipeline.Pipeline(steps=[('input', SimpleImputer(strategy='median')),
#                                           ('std_scaler', StandardScaler())])
#     data = pipeline.fit_transform(data)
#     test = np.array(test_country[2:]).reshape(1,-1)
#     pipe_test = pipeline['std_scaler'].transform(test)
#     return float(test[0][9].round(3))

# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[41]:


def q5():
    # Retorne aqui o resultado da questão 4.
    data = countries[['Net_migration']]
    q1,q3,x = data.quantile(0.25),data.quantile(0.75),1.5
    iqr = q3-q1
    low_outlier,high_outlier = q1 - x*iqr, q3 + x*iqr
    
    qtd_low,qtd_high = data[data < low_outlier].count(), data[data > high_outlier].count()
    
    return (int(qtd_low),int(qtd_high),False)
    


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[14]:


from sklearn.datasets import load_digits, fetch_20newsgroups

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[24]:


def q6():
    # Retorne aqui o resultado da questão 4.
    CV = CountVectorizer()
    freqs = CV.fit_transform(newsgroup.data)
    word_idx = CV.get_feature_names().index('phone')
    freqs_sum = freqs[:, word_idx].sum()
    
    return int(freqs_sum)
    
    


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[45]:


def q7():
    # Retorne aqui o resultado da questão 7.
    TV= TfidfVectorizer()
    tfidf = TV.fit_transform(newsgroup.data)
    word_idx = TV.get_feature_names().index('phone')
    freqs_sum = tfidf[:, word_idx].sum()
    
    return float(freqs_sum.round(3))
    
    

