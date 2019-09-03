import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
import quandl.api_config
import quandl.get_table
import seaborn as sns 
import scipy.optimize as sco

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
plt.style.use('fivethirtyeight')
np.random.seed(777)
quandl.ApiConfig.api_key='4KPQ9Y6_ND3VNqyfi7xc'
stocks = ['AAPL', 'AMZN','GOOGL','TSLA']
data = quandl.get_table('WIKI/PRICES', ticker = stocks, qopts ={'columns': ['date','ticker', 'adj_close']},
date={'gte':'2016-1-1', 'lte':'2017-12-31'}, paginate=True)
#rint (data.head())
#Let's we're going to transform the data
df=data.set_index('date')
table = df.pivot(columns='ticker')

#Retornos diarios
returns_daily = table.pct_change()
returns_annual = returns_daily.mean()*250

#Tenenomos los retornos diarios y la varainaza de las empresas
cov_daily = returns_daily.cov()
cov_annual = cov_daily*250# 250 numero de dias laborables

#Lista basias para almacenar los retonos volatilidad(Riesgo) valor imaginario.
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []
#Configuracion de numero de convinacion para el portafolio imaginario.
num_assets = len(stocks)
num_portfolios = 50000

#configurar secuencia de numerosmpara el portafolio 
np.random.seed(101) 

#Poblamos las listas basias con cada uno de los retornos, riesgos y valor de cada uno de los portafolio
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# definimos los valores del protfolio

portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

#Agregamos cada uno de los valores del portafolio para cada una de los stocks que ingresamos
for counter, symbol in enumerate(stocks): #number of stocks
    portfolio[symbol+' Weight']= [Weight[counter] for Weight in stock_weights]

df1 = pd.DataFrame(portfolio)
column_order = ['Returns','Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in stocks]
df1 = df1[column_order]


# plot frontier, max sharpe & min Volatility values with a scatterplot
#plt.style.use('fivethirtyeight')
#df1.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
 #               cmap='RdYlBu', edgecolors='black', figsize=(15, 8), grid=True)
#plt.xlabel('Volatility (Std. Deviation)')
#plt.ylabel('Expected Returns')
#plt.title('Efficient Frontier')
#plt.show()


#______________________________________________________________________________________________________________

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252 #Aqui me gustaria cambiarlo para hacer analisis de inversion trimestrales, Ya que los 252 es para anual
                                                 #creo yo que daria un mayor uso y un analisis mas rapido de poducto beta
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    #Aqui desarrollamos el portafolio con las diferentes alternativas
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(4)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

#completamos la tabla poniendo un riesgo minimo y un numero de portafolio
returns = table.pct_change()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_portfolios = 25000
risk_free_rate = 0.0100

#print (mean_returns.iloc[0])
#print (cov_matrix.iloc[0])

#la funcion que nos muestra la representacion grafica de la optimizacion del portafolio
def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)

    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=table.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=table.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    print ("\n")
    print (" Volatilidad Maxima del Portfolio\n")
    print ("Retorno Anual:", round(rp,2))
    print ("volatilidad anual:", round(sdp,2))
    print ("\n")
    print (max_sharpe_allocation)
    print ("\n")
    print (" Volatilidad minima del Portfolio \n")
    print ("Retorno Anula:", round(rp_min,2))
    print ("Volatilidad Anual:", round(sdp_min,2))
    print ("\n")
    print (min_vol_allocation)


    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='RdYlBu', marker='o', s=10, alpha=0.7)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='+',color='r',s=250, label='Optimzacion Maxima')
    plt.scatter(sdp_min,rp_min,marker='+',color='r',s=100, label='Optimizacion minima')

    plt.title(' Portafolio Optimizado') #Simulated Portfolio Optimization based on Efficient Frontier
    plt.xlabel('Volatilidad Anual')
    plt.ylabel('Retorno Anual')

    plt.show()

plt.show()
print (display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate))