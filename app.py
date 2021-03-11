import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def calc_metrics(y,y_pred):
  mse_naive = np.mean((y - y_pred)**2)
  rmse_naive = np.mean((y - y_pred)**2)
  mae = np.mean(np.abs(y - y_pred))

  return mse_naive, rmse_naive, mae

def transform_day(x, periodo):
  x = str(x)
  if periodo == 'Semana':
    if int(x[8:10]) <= 7:
      day = '3'
    elif int(x[8:10]) <= 14:
      day = '10'
    elif int(x[8:10]) <= 21:   
      day = '17'
    else:   
      day = '25'
    return x[:8]+day+x[10:] 
  if periodo == 'Mês':
    return x[:8]+'15'+x[10:]
  if periodo == 'Dia':
    return x

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

periodo = st.selectbox('Escolha o período de análise:', ['Dia','Semana','Mês'])

st.write("""## Falhas do {} por {}""".format(equipamento,periodo.lower()))

#ts = df[df['Equipamento'] == equipamento].groupby('Data').count().reset_index().drop(["Equipamento"],axis=1)
#ts = ts.rename(columns={'Classe':'Falhas'})
#st.line_chart(ts.rename(columns={'Data':'index'}).set_index('index'))

#st.write("""## Falhas do {} por mês""".format(equipamento))

#df['Data'] = df['Data'].apply(lambda x: str(x))
#df['Data'] = df['Data'].apply(lambda x: x[:8]+'15'+x[10:])
df['Data'] = df['Data'].apply(lambda x: transform_day(x,periodo))
df['Data'] = df['Data'].astype("datetime64")

ts = df[df['Equipamento'] == equipamento].groupby('Data').count().reset_index().drop(["Equipamento"],axis=1)
ts = ts.rename(columns={'Classe':'Falhas'})
st.line_chart(ts.rename(columns={'Data':'index'}).set_index('index'))

media = ts.mean()

st.write("""### Mínimo de falhas por {}: {}""".format(periodo.lower(),ts.min()['Falhas']))
st.write("""### Média de falhas por {}: {}""".format(periodo.lower(),ts.mean()['Falhas']))

st.write("""## Modelo Naive""")

ts = ts.set_index(['Data'])

naive_model = ts.shift().dropna()

metrics = calc_metrics(ts['Falhas'].values[1:],ts.shift().dropna()['Falhas'].values)

st.write("""### MSE = {}""".format(metrics[0]))
st.write("""### RMSE = {}""".format(metrics[1]))
st.write("""### MAE = {}""".format(metrics[2]))


plt.plot(naive_model,label='Naive')
plt.plot(ts,label='Dados')
#plt.legend()
plt.ylabel('Falhas')
plt.xlabel('Data')
plt.legend()
st.pyplot(plt)

st.write("""## Teste de Dickey-Fuller Aumentado""")

dftest = adfuller(ts['Falhas'],autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-Value','Lags Used','Observations'])
resultado = 'estacionária' if dfoutput[1] < 0.05 else 'não estacionária'
st.write(dfoutput)
st.write('### A série é {}'.format(resultado))

st.write("""## Modelo ARIMA""")

from statsmodels.tsa.arima.model import ARIMA

p = st.slider('P', min_value=0, max_value=10)
d = st.slider('d', min_value=0, max_value=10)
q = st.slider('Q', min_value=0, max_value=10)

model = ARIMA(ts,order=(p,d,q))
results_AR = model.fit()

metrics = calc_metrics(ts['Falhas'].values,results_AR.fittedvalues.values)

st.write("""### MSE = {}""".format(metrics[0]))
st.write("""### RMSE = {}""".format(metrics[1]))
st.write("""### MAE = {}""".format(metrics[2]))

plt.clf()
plt.plot(results_AR.fittedvalues,label='ARIMA')
plt.plot(ts,label='Dados')
#plt.legend()
plt.ylabel('Falhas')
plt.xlabel('Data')
plt.legend()
st.pyplot(plt)

st.write("""## Previsão""")

#results_AR.plot_predict(1,100) 

artigo = {"Mês":"o","Semana":"a","Dia":"o"}

mes = st.selectbox('Escolha {} {}:'.format(artigo[periodo],periodo.lower()), [i for i in range(1,13)])

previsao = results_AR.forecast(steps=mes).values[-1]

st.write(previsao)

avaliacao = "acima" if previsao >= ts.mean().values else "abaixo"

st.write("""### O número de falhas d{} {} {} está {} da média""".format(artigo[periodo],periodo.lower(),mes,avaliacao))


