import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#숫자 데이터 받아오기
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test,y_test) = mnist.load_data()

x_train, y_train = x_train/255.0, y_train/255.0 #0~255값을 가진 데이터를 0~1로 정규화 시킨다.
print(x_train.shape) #28,28데이터로 주어졌다. 이를 컨벤셔널로 이용해 특징을 추출해낸다.
x_train=x_train.reshape(-1,28,28,1)
print(y_train.shape)
y_train=to_categorical(y_train)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),       # adding convolution layer with input size (28,28,1) , 1 means the images are in greyscale not rgb
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),                                              # adding pooling layer
        tf.keras.layers.Dropout(0.25),
    
        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),  
        tf.keras.layers.Conv2D(192,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
    
    
        tf.keras.layers.Flatten(),                                                      # flatten will flatten the input (28,28,1) to a single array
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512,activation='relu'),                                   # hidden layer with 256 units
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10,activation='softmax')  
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=20)

model.evaluate(x_test,y_test)

