#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[ ]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[4]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return int(black_friday[(black_friday['Gender']== 'F') & (black_friday['Age'] == '26-35')]['Age'].count())
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[5]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return int(black_friday['User_ID'].nunique())
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[6]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return int(black_friday.dtypes.nunique())
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[7]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return float(1 - black_friday.dropna(how='any').shape[0]/black_friday.shape[0])
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[8]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return int(black_friday.isna().sum().max())
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[9]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].dropna().mode().loc[0]
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[10]:


def q8():
    # Retorne aqui o resultado da questão 8.
    x = black_friday['Purchase'].to_numpy().reshape(-1,1)
    return float(MinMaxScaler().fit_transform(x).mean())
    pass

# usei inicialmente float((black_friday['Purchase']/black_friday['Purchase'].max()).mean()). 
# O resultado deu aproximado (0.39), mas percebi via Code Review que usando o pacote sklearn obteria
# o resultado certo para o teste.


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[11]:


def q9():
    # Retorne aqui o resultado da questão 9.
    bf = black_friday
    bf['Purchase'] = (bf['Purchase'] - bf['Purchase'].mean())/bf['Purchase'].std()
    bf = bf[(bf['Purchase'] >= -1) & (bf['Purchase'] <= 1)]
    return int(bf['Purchase'].count())
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[12]:


def q10():
    bf = black_friday[(black_friday['Product_Category_2'].isna() == True)]
    return bf['Product_Category_2'].equals(bf['Product_Category_3'])
    # Retorne aqui o resultado da questão 10.
    pass

