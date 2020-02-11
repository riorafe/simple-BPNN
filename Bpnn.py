import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# LOAD DATASET
def load_dataset():
    data = pd.read_csv("LifeExpectancy.csv") # return nya DataFrame
    x_data = data[ ["Gender", "Residential" , "Physical Activity (times per week)", "Happiness"] ]
    y_data = data[ ["Life Expectancy"] ]
    return x_data, y_data

def predict():
    # LINEAR COMBINATION

    # INPUT -> HIDDEN
    Wx1_b = tf.matmul(x, weight["hidden"]) + bias["hidden"]
    y1 = tf.nn.sigmoid(Wx1_b)

    # HIDDEN1 -> HIDDEN2

    # HIDDEN -> OUTPUT
    Wx2_b = tf.matmul(y1, weight["output"]) + bias["output"]
    y2 = tf.nn.sigmoid(Wx2_b)

    return y2

x_data, y_data = load_dataset()

# CONVERT STRING FEATURES TO NUMERIC

# 1. Bikin Object Encoder
ordinal_encoder = OrdinalEncoder() # Optional kalo inputnya ada yang string

# 2. Fit encoder ke data
ordinal_encoder = ordinal_encoder.fit(x_data[ ["Gender", "Residential"] ]) # Karena cuman mau ubah yang gender dan Residential ke angka (mapping)

# 3. Transform datanya
x_data[ ["Gender", "Residential"] ] = ordinal_encoder.transform(x_data[ ["Gender", "Residential"] ])

# Bisa langsung pakai .fit_transform
# x_data[ ["Gender", "Residential"] ] = ordinal_encoder.fit_transform(x_data[ ["Gender", "Residential"] ])


# ONE HOT ENCODING FOR THE TARGET

# SPARSE = FALSE SUPAYA RETURN NYA ARRAY, KALO TRUE JADI SPARSE MATRIX
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder = one_hot_encoder.fit(y_data)
y_data = one_hot_encoder.transform(y_data)

# Bisa langsung pakai .fit_transform
# y_data = one_hot_encoder.fit_transform(y_data)

# SPLIT DATASET INTO TRAINING & TESTING SET

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)

# NORMALIZE DATASET

scaler = MinMaxScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# CREATE MODEL ARCHITECTURE

layer = {
    'num_input' : 4,
    'num_hidden' : 4,
    # 'num_hidden2' : 4,
    'num_output' : 3
}

weight = {
    'hidden' : tf.Variable(tf.random_normal( [ layer["num_input"], layer["num_hidden"] ] )),
    'output' : tf.Variable(tf.random_normal( [ layer["num_hidden"], layer["num_output"] ] ))
}

bias = {
    'hidden' : tf.Variable(tf.random_normal( [ layer["num_hidden"] ] )),
    'output' : tf.Variable(tf.random_normal( [ layer["num_output"] ] ))
}

# CREATE PLACEHOLDER FOR FEEDING DATA
x = tf.placeholder(tf.float32, [None, layer["num_input"] ])
y_true = tf.placeholder(tf.float32, [None, layer["num_output"] ])

y_predict = predict()

# VARIABLE FOR TRAINING
learning_rate = 0.1
number_of_epoch = 1000

# LOSS FUNCTION & OPTIMIZER
# MEAN SQUARED ERROR
loss = tf.reduce_mean(0.5 * (y_true - y_predict) ** 2) # cari rata-rata setelah cari lossnya
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

# TRAINING THE MODEL

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1, number_of_epoch + 1):
        sess.run(train, feed_dict = {
            x : x_train,
            y_true : y_train
        })
        loss_val = sess.run(loss, feed_dict = {
            x : x_train,
            y_true : y_train
        })

        print("Epoch: {}, Loss: {}".format(i, loss_val))

# TESTING THE MODEL

with tf.Session() as sess:
    sess.run(init)

    # HITUNG AKURASI DARI MODEL
    matches = tf.equal(tf.argmax(y_predict, axis = 1), tf.argmax(y_true, axis = 1))
    
    # Matches returnnya boolean = [true, true, false, true, false]
    accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

    accuracy_val = sess.run(accuracy * 100, feed_dict = {
        x : x_test,
        y_true : y_test
    })

    print("Accuracy = ", accuracy_val)

