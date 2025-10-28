import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tenflow.keras.models import load_model

df  = pd.read_csv('data/customer_churn.csv')

X = df.drop(['CustomerId','Surname','Exited'], axis=1)
y = df['Exited']

X['Geography'] = LabelEncoder().fit_transform(X['Geography'])
X['Gender'] = LabelEncoder().fit_transform(X['Gender'])

#tranform of the data 
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(16, activation ='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')

])

model.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 30,batch_size=13, validation_split = 0.2, verbose = 1)



model.save('models/churn_model.h5')
pd.to_pickle(scaler,'.model/scler.pkl')



