import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.write("""# Séries Temporais Para Previsão de Falhas""")

df=pd.read_excel('manutencaoexcel.xlsx')
df = df[df['Classe']=='CORRETIVA']
df['Classe'] = 1
st.write("""## 15 Equipamentos com maior número de falhas""")
st.write(df.groupby('Equipamento').count().reset_index().sort_values('Classe',ascending=False).head(15))

equipamentos = []
for equipamento in df.groupby('Equipamento').count().reset_index().sort_values('Classe',ascending=False).head(15)['Equipamento'].iloc:
    equipamentos.append(equipamento)

equipamento = st.selectbox('Escolha o equipamento:', equipamentos)

st.write("""## Falhas do {} por dia""".format(equipamento))

ts = df[df['Equipamento'] == equipamento].groupby('Data').count().reset_index().drop(["Equipamento"],axis=1)
ts = ts.rename(columns={'Classe':'Falhas'})
st.line_chart(ts.rename(columns={'Data':'index'}).set_index('index'))

st.write("""## Falhas do {} por mês""".format(equipamento))

df['Data'] = df['Data'].apply(lambda x: str(x))
df['Data'] = df['Data'].apply(lambda x: x[:8]+'15'+x[10:])
df['Data'] = df['Data'].astype("datetime64")

ts = df[df['Equipamento'] == equipamento].groupby('Data').count().reset_index().drop(["Equipamento"],axis=1)
ts = ts.rename(columns={'Classe':'Falhas'})
st.line_chart(ts.rename(columns={'Data':'index'}).set_index('index'))
st.write('Mínimo de falhas por mês: {}'.format(ts.min()['Falhas']))
