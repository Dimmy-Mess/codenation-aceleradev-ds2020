import streamlit as st
import pickle
import time
import numpy as np
import pandas as pd
import seaborn as sns
import base64
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split



def get_infos(result, mkt):
    return mkt.loc[mkt['id'].isin(result['id_result']),:]


def prepare_dataset(data):
	df = pd.read_csv('1.pre-processamento/df_preprocessed.csv')
	new = df.loc[df['id'].isin(data['id']), df.columns]
	new.set_index('id', inplace=True)
	return new


def neighbors_lead_generator(nn, df_p):

	index_list = np.array([])
	df = pd.read_csv('1.pre-processamento/df_preprocessed.csv')
	df = df.set_index('id')
	
	for i in range(df_p.shape[0]):
		k_distances, k_indexes = nn.kneighbors(df_p.iloc[[i]])
		k_indexes = np.delete(k_indexes, [0])
		index_list = np.concatenate((index_list, k_indexes), axis=None)

	neighbors = []

	for i in range(len(index_list)):
		neighbors.append(df.iloc[int(index_list[i])].name)

	lead = pd.DataFrame(neighbors, columns=['id_result'])
	lead.drop_duplicates(keep='first', inplace=True, ignore_index=True)

	return lead

def get_table_download_link(df):

	csv = df.to_csv(index=False)
	b64 = base64.b64encode(
    csv.encode()
	).decode()  # some strings <-> bytes conversions necessary here
	return f'<a href="data:file/csv;base64,{b64}" download="results.csv">Faça o Download de seus resultados!</a>'

with open('2.treinamento/n_neighbors.pikle','rb') as f:
    nn = pickle.load(f)
    



st.markdown("# Projeto Prático AceleraDev DataScience 2020")
st.markdown("## Objetivo")
st.markdown("O objetivo deste produto é fornecer um serviço automatizado que recomenda leads para um usuário dado sua atual lista de clientes (Portfólio).")

st.markdown('Caso não tenha os portfólios faça o download nos links abaixo:')
st.markdown('* [Portfólio 1](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio1.csv)')
st.markdown('* [Portfólio 2](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio2.csv)')
st.markdown('* [Portfólio 3](https://codenation-challenges.s3-us-west-1.amazonaws.com/ml-leads/estaticos_portfolio3.csv)')
st.markdown('_________________')


st.markdown('# Encontre Clientes com base em seu Portfólio')
st.set_option('deprecation.showfileUploaderEncoding', False)

df_p = st.file_uploader('Insira o portfolio em formato csv',type=('csv'))



if df_p != None:
	st.markdown('**Muito bem!** O algoritmo trabalha por similaridade. Selecionamos clientes em potencial com base em cada cliente de seu portfólio!')
	st.markdown('Quando estiver pronto, clique no botão "CALCULAR!" abaixo. O processo inteiro deve levar entre 5 e 10 minutos.')
	calc = st.button('CALCULAR!')
	
	if calc:
		start = time.time()
		
		st.markdown('Preparando Dataset de seu Portfólio...')
		df_p = pd.read_csv(df_p)
		df_p = prepare_dataset(df_p)
		
		st.markdown('Preparando Resultados...')
		results = neighbors_lead_generator(nn,df_p)
		
		end = time.time()
		st.markdown(f'Tempo gasto: {end-start}')
		
		st.markdown('______________________')
		st.markdown('# Resultados')
		
		st.markdown(f'Encontramos *{results.shape[0]} novos clientes* para você!')
		st.markdown('Confira:')
		
		mkt = pd.read_csv('csv/estaticos_market/estaticos_market.csv')[['id','setor','de_ramo','sg_uf']]
		results = get_infos(results,mkt)
				
		st.markdown('### Setores')
		sns.countplot(data=results,y='setor', palette='Set3')
		st.pyplot()
		
		st.markdown('### Ramo de atividade')
		
		sns.countplot(data=results,y='de_ramo', palette='Set3')
		st.pyplot()
		
		st.markdown('### Estados')
		
		sns.countplot(data=results,y='sg_uf', palette='Set3')
		st.pyplot()
		
		st.markdown(get_table_download_link(results), unsafe_allow_html=True)
		
