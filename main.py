from sklearn import preprocessing
from collections import deque
import pandas as pd
import numpy as np 
import random 
import time

main_df = pd.read_csv("/content/BTC-USD.csv", names = ['time','high','low','open','close','vol'])
# print(main_df.head())

# we would be deviding the data on the bases of time so set time as index
main_df.set_index('time',inplace = True)

# we want to predict the price 3 steps ahead 
Future_prediction_step = 3
main_df['future'] = main_df["close"].shift(-Future_prediction_step)

# to make tagets we will write a func to classify raise as 1 and decline as 0 in price
def classify(current,future):
  if float(current) < float(future):
    return 1
  else:
    return 0

main_df.fillna(method = 'ffill', inplace = True)
main_df.dropna(inplace = True)    

main_df["target"] = list(map(classify,main_df["close"],main_df["future"]))
main_df.drop("future",1)

main_df.dropna(inplace = True)

# now we will split the data in traning and validation data
times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

# now we will wrtie a func to preprocess train and validation data 
def preprocess_df(df):

  for col in df.columns:
    if col != "target":
      df[col] = df[col].pct_change()
      df.dropna(inplace = True)
      df[col] = preprocessing.scale(df[col].values)
      df.dropna(inplace = True)
    
  seqlen = 60
  sequential_data = []
  prev_days = deque(maxlen = seqlen) 

  for i in df.values:
    prev_days.append([n for n in i[:-1]])
    if len(prev_days) == seqlen:
      sequential_data.append([np.array(prev_days),i[-1]]) 

  inc,dec = [],[]

  for seqdata,target in sequential_data:
    if target == 1:
      inc.append([np.array(seqdata),target]) 
    elif target == 0:
      dec.append([np.array(seqdata),target])  

  lowkey = min(len(inc),len(dec))   
  inc,dec = inc[:lowkey],dec[:lowkey]  

  sequential_data = inc + dec
  random.shuffle(sequential_data)  

  X,Y = [],[]
  for seq,target in sequential_data:
    X.append(seq.tolist())
    Y.append(target)

  return np.array(X),np.array(Y)
  
train_x,train_y = preprocess_df(main_df)
validation_x,validation_y = preprocess_df(validation_df)

# print(f"training samples: {len(train_x)} validation samples: {len(validation_x)}")

#libs to train the model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

model = Sequential()

model.add(LSTM(128,input_shape = train_x.shape[1:],return_sequences = True))
model.add(Dropout(.2))
model.add(BatchNormalization())

model.add(LSTM(128,return_sequences = True))
model.add(Dropout(.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(.2))
model.add(BatchNormalization())

model.add(Dense(32,activation = 'relu'))
model.add(Dropout(.2))

model.add(Dense(2, activation = 'softmax'))


opt = Adam(lr = 0.01, decay = 1e-6)
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = opt, metrics = ['accuracy'])

history = model.fit(train_x, train_y, batch_size = 64,epochs = 5, validation_data = (validation_x, validation_y))

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save()
