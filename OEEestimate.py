# %%
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow import keras
#ML Approach to oee
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# %%
data = pd.read_csv("f:/oee data1.csv", engine='python',encoding='latin1')
# %%
data.head()
data['OEE']=data['OEE']*100
data['Quality']=data['Quality']*100
data['Performance']=data['Performance']*100
data['Availability']=data['Availability']*100
# %%

data['mThroghput']=60/data['Cycle Time ']

data['mThroghput'].corr(data['Throughput '])
data['Throughput 1'] = data['Throughput '].iloc[:].rolling(window=90).mean()
data['Throughput 2'] = data['mThroghput'].iloc[:].rolling(window=90).mean()
plt.plot(data['Throughput 1'],label='SMA 3 Months')


plt.plot(data['Throughput 2'],label='SMA 3 Months')
plt.legend

pd.plotting.scatter_matrix(data[['Quality','Performance','Availability','OEE']], hist_kwds={'bins':15},figsize=(10,10),color='darkblue')
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
boxplot = sns.boxplot(data=data,x='OEE',ax=ax[0])
boxplot1 = sns.boxplot(data=data,x='Throughput ',ax=ax[1])

num_coln = data.select_dtypes(include=np.number).columns.tolist()
bins=10
j=1
fig = plt.figure(figsize = (20, 30))
for i in num_coln:
    plt.subplot(7,4,j)
    plt.boxplot(data[i])
    j=j+1
    plt.xlabel(i)
    plt.legend(i)
plt.show()
# %%
cor = data.corr()
plt.figure(figsize = (27,26))
sns.heatmap(cor, annot = True, cmap = 'coolwarm')
plt.show()
X = data.drop(['OEE','Quality','Performance','Availability'])
y = data['OEE']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


#training deep learning approaches using k-fold cross-validation
# %%
import numpy as np
k=5
num_val_samples = len(X_train) // k
num_epochs = 1000
all_scores = []
def build_model():
    model = Sequential()
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer, loss='mse',metrics=['mae'])
    return model
n=500
all_mae_histories = []
all_mae_histories1 = []
for i in range(k):
    print('processing fold #', i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([X_train[:i * num_val_samples],X_train[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([y_train[:i * num_val_samples],y_train[(i + 1) * num_val_samples:]],axis=0)
    model = build_model()
    history=model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=n, batch_size=64, verbose=0)
    val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    mae_history = history.history['val_loss']
    mae_history1=history.history['loss']
    all_mae_histories.append(mae_history)
    all_mae_histories1.append(mae_history1)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(n)]
average_mae_history1 = [np.mean([x[i] for x in all_mae_histories1]) for i in range(n)]
# %%
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.plot(range(1, len(average_mae_history1) + 1), average_mae_history1)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


from sklearn.metrics import mean_squared_error,mean_absolute_error
#test the deep neural network model on the test data 
predictions = model.predict(X_test)
#test the model with mae metric , for nmae you should dvide it with the avarage of y_test
print('mae:',mean_absolute_error(y_test,predictions))
#test the model with rmse metric , for nrmse you should dvide it with the avarage of y_test
print('rmse:',np.sqrt(mean_squared_error(y_test,predictions)))
#test the model with r2
dnn_preds = model.predict(X_test).ravel()
print('R2:',r2_score(y_test,  dnn_preds))

# the LSTM approach
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train, y_train = np.array(X_train), np.array(y_train)
X_test,y_test = np.array(X_test), np.array(y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
k=5
num_val_samples = len(X_train) // k
num_epochs = 1000
all_scores = []
def build_model():
    regressor = Sequential()


# Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 200, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 100, return_sequences = True))
    regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

# Adding the output layer
    regressor.add(Dense(units = 1))

# Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor
n=1000
all_mae_histories = []
all_mae_histories1 = []
for i in range(k):
    print('processing fold #', i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([X_train[:i * num_val_samples],X_train[(i + 1) * num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([y_train[:i * num_val_samples],y_train[(i + 1) * num_val_samples:]],axis=0)
    regressor = build_model()
    history=regressor.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=n, batch_size=64, verbose=0)
    mae_history = history.history['val_loss']
    mae_history1=history.history['loss']
    val_mae = regressor.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    all_mae_histories.append(mae_history)
    all_mae_histories1.append(mae_history1)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(n)]
average_mae_history1 = [np.mean([x[i] for x in all_mae_histories1]) for i in range(n)]
# %%
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.plot(range(1, len(average_mae_history1) + 1), average_mae_history1)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

from sklearn.metrics import mean_squared_error,mean_absolute_error
#test the deep neural network model on the test data 
predictions = model.predict(X_test)
#test the model with mae metric , for nmae you should dvide it with the avarage of y_test
print('mae:',mean_absolute_error(y_test,predictions))
#test the model with rmse metric , for nrmse you should dvide it with the avarage of y_test
print('rmse:',np.sqrt(mean_squared_error(y_test,predictions)))
#test the model with r2
dnn_preds = model.predict(X_test).ravel()
print('R2:',r2_score(y_test,  dnn_preds))

# %%
#mae cross
import numpy as np 
import matplotlib.pyplot as plt 
  
X = ['Fold 1','Fold 2','Fold 3','Fold 4','Fold 5']

  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.1, DNN, 0.2, label = 'DNN')
plt.bar(X_axis + 0.1, LSTM, 0.2, label = 'LSTM')
  
plt.xticks(X_axis, X)
plt.xlabel("Models with 5 Fold Cross-validation")
plt.ylabel("MAE")
plt.title("Models MAE evaluation on each Fold")
plt.legend()
plt.show()
#rmse cross
import numpy as np 
import matplotlib.pyplot as plt 
  

X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.1, DNN, 0.2, label = 'DNN')
plt.bar(X_axis + 0.1, LSTM, 0.2, label = 'LSTM')
  
plt.xticks(X_axis, X)
plt.xlabel("Models with 5 Fold Cross-validation")
plt.ylabel("RMSE")
plt.title("Models RMSE evaluation on each Fold")
plt.legend()
plt.show()

#regression plot with disturbiotion and pearson corrolation
df1 = pd.DataFrame(y_test.values, columns=[ 'Actual'])
df2 = pd.DataFrame(dnn_preds, columns=['Prediction'])
aw=df1['Actual']
aw1=df2['Prediction']
import scipy.stats as stats
graph=sns.jointplot(x =aw , y = aw1, kind='kde', space=0, fill=True,color='blue')
#pearson corrolation analyis on the test data
pearson_corr, _ = stats.pearsonr(y_test, dnn_preds)
# if you choose to write your own legend, then you should adjust the properties then
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
# here graph is not a ax but a joint grid, so we access the axis through ax_joint method
graph.ax_joint.legend([phantom],['pearson_corr={:f}'.format(pearson_corr)])