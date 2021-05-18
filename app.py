#importando as bibliotecas
import streamlit as st #pacote para fazer o app
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet

#função para calcular métricas
#entradas: a série temporal original (y)
#          o modelo (y_pred)
#saida: métricas 
def calc_metrics(y,y_pred):
  mse = np.mean((y - y_pred)**2)#média da diferença ao quadrado
  rmse = np.sqrt(mse)
  mae = np.mean(np.abs(y - y_pred))#média da diferença absoluta
  mean = np.mean(y)
  #st.write(mean)
  r2 = 1 - (np.sum((y - y_pred)**2))/(np.sum((y - mean)**2))
  mape = (1/len(y))*np.sum(np.abs((y-y_pred)/y))
  smape = (1/len(y))*np.sum(np.abs((y-y_pred))/(np.abs(y)+np.abs(y_pred)))

  st.write("""### MSE = {}""".format(mse))
  st.write("""### RMSE = {}""".format(rmse))
  st.write("""### MAE = {}""".format(mae))
  st.write("""### MAPE = {}""".format(mape))
  st.write("""### SMAPE = {}""".format(smape))
  st.write("""### $R^2$ = {}""".format(r2))

  #return mse, rmse, mae, r2

#função para transformar as datas, serve para depois agrupar em semana ou mes
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

#//--------------------------------------------------------------------------------------------------------------------------//

#o comando st.write escreve uma mensagem no app
st.write("""# Séries Temporais Para Previsão de Falhas""")

#leitura do arquivo
df=pd.read_excel('manutencaoexcel.xlsx')
#considerando apenas manutenções corretivas
df = df[df['Classe']=='CORRETIVA']
#considerar cada manutenção como uma "falha"
df['Classe'] = 1
st.write("""## 15 Equipamentos com maior número de falhas""")
#agrupando por equipamento, oredenando por ordem decrescente, pegando apenas os 15 primeiros
st.write(df.groupby('Equipamento').count().reset_index().sort_values('Classe',ascending=False).head(15))

#iniciando lista de equipamentos vazia
equipamentos = []
#loop para adicionar na lista os 15 equipamentos com o maior número de falhas
for equipamento in df.groupby('Equipamento').count().reset_index().sort_values('Classe',ascending=False).head(15)['Equipamento'].iloc:
    equipamentos.append(equipamento)

#cria caixa de seleção com a lista de equipamentos
equipamento = st.selectbox('Escolha o equipamento:', equipamentos)
#cria caixa de seleção de período
periodo = st.selectbox('Escolha o período de análise:', ['Dia','Semana','Mês'])
#mensagem na tela dinâmica, varia com o equipamento e período escolhidos
st.write("""## Falhas do {} por {}""".format(equipamento,periodo.lower()))
#aplica a transformação nas datas para agrupar por período
df['Data'] = df['Data'].apply(lambda x: transform_day(x,periodo))
df['Data'] = df['Data'].astype("datetime64")
#agrupamento por período e salvando o dataset em ts 
ts = df[df['Equipamento'] == equipamento].groupby('Data').count().reset_index().drop(["Equipamento"],axis=1)
ts = ts.rename(columns={'Classe':'Falhas'})
#cópia para o modelo prophet
ts_prophet = ts.copy()
#plot do dataset já agrupado
st.line_chart(ts.rename(columns={'Data':'index'}).set_index('index'))
#cálculo da média
media = ts.mean()
#imprime o número mínimo de falha e média
st.write("""### Mínimo de falhas por {}: {}""".format(periodo.lower(),ts.min()['Falhas']))
st.write("""### Média de falhas por {}: {}""".format(periodo.lower(),ts.mean()['Falhas']))

#//--------------------------------------------------------------------------------------------------------------------------//

st.write("""## Teste de Dickey-Fuller Aumentado""")

dftest = adfuller(ts['Falhas'],autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-Value','Lags Used','Observations'])
resultado = 'estacionária' if dfoutput[1] < 0.05 else 'não estacionária'
st.write(dfoutput)
st.write('### A série é {}'.format(resultado))

#//--------------------------------------------------------------------------------------------------------------------------//

st.write("""## Modelos""")

escolha_modelo = st.selectbox("""# Escolha o modelo:""", ['Naive','ARIMA','Prophet'])
ts = ts.set_index(['Data'])
if escolha_modelo == 'Naive':

  #seção do modelo naive
  st.write("""### Modelo Naive""")
  #passa data como índice
  #o módelo naive é a série temporal deslocada
  naive_model = ts.shift().dropna()

  #paramos aqui -----------------------------------------------------------------------------------------------------------////////////////////////////////////////
  metrics = calc_metrics(ts['Falhas'].values[1:],ts.shift().dropna()['Falhas'].values)

  #st.write("""### MSE = {}""".format(metrics[0]))
  #st.write("""### RMSE = {}""".format(metrics[1]))
  #st.write("""### MAE = {}""".format(metrics[2]))


  plt.plot(naive_model,label='Naive')
  plt.plot(ts,label='Dados')
  #plt.legend()
  plt.ylabel('Falhas')
  plt.xlabel('Data')
  plt.legend()
  st.pyplot(plt)

#//--------------------------------------------------------------------------------------------------------------------------//
if escolha_modelo == 'ARIMA':
  st.write("""### Modelo ARIMA""")

  p = st.slider('P', min_value=0, max_value=11)
  d = st.slider('d', min_value=0, max_value=10)
  q = st.slider('Q', min_value=0, max_value=10)

  model = ARIMA(ts,order=(p,d,q))
  results_AR = model.fit()

  metrics = calc_metrics(ts['Falhas'].values,results_AR.fittedvalues.values)

  #st.write("""### MSE = {}""".format(metrics[0]))
  #st.write("""### RMSE = {}""".format(metrics[1]))
  #st.write("""### MAE = {}""".format(metrics[2]))

  plt.clf()
  plt.plot(results_AR.fittedvalues,label='ARIMA')
  plt.plot(ts,label='Dados')
  #plt.legend()
  plt.ylabel('Falhas')
  plt.xlabel('Data')
  plt.legend()
  st.pyplot(plt)

#//--------------------------------------------------------------------------------------------------------------------------//
if escolha_modelo == 'Prophet':
  st.write("""### Modelo Prophet""")

  ts_prophet = ts_prophet.rename(columns={'Falhas':'y','Data':'ds'})

  model = Prophet()
  model.fit(ts_prophet)

  predictions = model.predict(ts_prophet)[['ds','yhat']]
  predictions = predictions.set_index(['ds'])
  metrics = calc_metrics(ts['Falhas'].values,predictions['yhat'].values)

  #st.write("""### MSE = {}""".format(metrics[0]))
  #st.write("""### RMSE = {}""".format(metrics[1]))
  #st.write("""### MAE = {}""".format(metrics[2]))

  plt.clf()
  plt.plot(predictions,label='Prophet')
  plt.plot(ts,label='Dados')
  plt.legend()
  plt.ylabel('Falhas')
  plt.xlabel('Data')
  plt.legend()
  st.pyplot(plt)

st.write("""## Avaliação considerando treino e teste""")

porcentagem = st.selectbox('Escolha o percentual da base de teste:', ['0.05','0.1','0.25','0.5','0.75'])

#st.write(len(ts))
#st.write(len(ts_prophet))
numero_teste = int(len(ts)*float(porcentagem))

treino_arima = ts[:-numero_teste]
teste_arima = ts[-numero_teste:]

#arima_model = auto_arima(treino_arima,error_Action='warn',supress_warnings=True,stepwise=True)

#forecast_arima = arima_model.predict(n_periods=numero_teste)

model = ARIMA(treino_arima,order=(5,0,5))
results_AR = model.fit()

#metrics = calc_metrics(teste_arima['Falhas'].values,forecast_arima)
#teste_arima['Falhas'] = forecast_arima

metrics = calc_metrics(teste_arima['Falhas'].values,results_AR.forecast(steps=numero_teste).values)

teste_arima['Falhas'] = results_AR.forecast(steps=numero_teste).values

ts_prophet = ts_prophet.rename(columns={'Falhas':'y','Data':'ds'})

treino_prophet = ts_prophet[:-numero_teste]
teste_prophet = ts_prophet[-numero_teste:]

prophet_model = Prophet()
prophet_model.fit(treino_prophet)
forecast_prophet = prophet_model.predict(teste_prophet)
metrics = calc_metrics(teste_prophet['y'].values,forecast_prophet['yhat'].values)
teste_prophet['y'] = forecast_prophet['yhat']

teste_prophet['y'] = np.mean(treino_prophet['y'])
metrics = calc_metrics(teste_prophet['y'].values,forecast_prophet['yhat'].values)

plt.clf()
plt.plot(ts,label='Dados')
plt.plot(teste_arima,label='Arima')
plt.plot(forecast_prophet['ds'],forecast_prophet['yhat'],label='Prophet')
plt.legend()
plt.ylabel('Falhas')
plt.xlabel('Data')
plt.legend()
st.pyplot(plt)

st.write("""## Previsão""")

#results_AR.plot_predict(1,100) 

artigo = {"Mês":"o","Semana":"a","Dia":"o"}

mes = st.selectbox('Escolha {} {}:'.format(artigo[periodo],periodo.lower()), [i for i in range(1,13)])

intervalo = st.selectbox('Escolha o intervalo de confiança (%):', [0.95,0.9,0.85,0.8])

model = ARIMA(ts,order=(5,0,5))
results_AR = model.fit()
previsao = results_AR.forecast(steps=mes)
previsao = previsao.values[-1]
intervalo = results_AR.conf_int((1-intervalo)/100)

st.write(previsao)

avaliacao = "acima" if previsao >= ts.mean().values else "abaixo"
st.write(intervalo)
st.write("""### O número de falhas d{} {} {} está {} da média""".format(artigo[periodo],periodo.lower(),mes,avaliacao))



