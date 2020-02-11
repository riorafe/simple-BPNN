import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

data=pd.read_csv('LifeExpectancy.csv')

x_data = data[['Gender', 'Residential', 'Physical Activity (times per week)', 'Happiness']]
y_data = data[['Life Expectancy']]

x_data[['Gender', 'Residential']] = OrdinalEncoder().fit_transform(x_data[['Gender', 'Residential']])
y_data = OneHotEncoder(sparse=False).fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()

col=x_train.shape[1]

model.add(Dense(1000, activation='relu', input_shape =(col,)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.2, epochs = 30)

model.evaluate(x_test, y_test, verbose=2)