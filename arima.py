from numpy.core.shape_base import block
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
df=pd.read_csv('audaxlabs/abc.csv')
#print(df)
values = df['Open'].values
df["time"] = df.index
SPLIT_SIZE = .8
v1 = values[0:int(len(values)*0.8)]
v2 =values[int(len(values)*0.8):len(values)]
print(v1)
hist = [item for item in v1]
predict= []
for item in v2:
    model = ARIMA(hist,order=(1,1,2))
    
    cool = hist[-1]
   
    model1 = model.fit()
    thing = model1.forecast()[0]
    predict.append(thing)
    print(f'{thing},{cool}')
    #input()
    hist.append(item)
plt.plot(range(0,len(v2)),v2)
plt.plot(range(0,len(v2)),predict,color="red")
plt.show()



#print(model1.summary())
