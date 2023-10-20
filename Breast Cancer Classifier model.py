# Import required dependencies
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import numpy as np


# Load the dataset
cancer_dataset = sklearn.datasets.load_breast_cancer()
cancer_dataset_dataFrame = pd.DataFrame(cancer_dataset.data,columns=cancer_dataset.feature_names)
cancer_dataset_dataFrame['label']=cancer_dataset.target
# Show the first 5 rows in the dataset
cancer_dataset_dataFrame.head()
# Show the last 5 rows in the dataset
cancer_dataset_dataFrame.tail()
# Show the shape of the dataset
cancer_dataset_dataFrame.shape
# Find the relation between the output label column and all dataset features
cancer_dataset_dataFrame.groupby('label').mean()
# Find some statistical info about the dataset
cancer_dataset_dataFrame.describe()



# Plot the distribution of all dataset featuers
columns_names = cancer_dataset_dataFrame.columns
package_colors=sns.color_palette('husl',n_colors=len(columns_names))
for i,column_name in enumerate(columns_names):
    plt.figure(figsize=(5,5))
    sns.distplot(x=cancer_dataset_dataFrame[column_name],color=package_colors[i])

# Count the groups in the output column and its repetition
plt.figure(figsize=(5,5))
sns.countplot(x = 'label',data=cancer_dataset_dataFrame)
cancer_dataset_dataFrame['label'].value_counts()



# Split the dataset into input and label data
X = cancer_dataset_dataFrame.drop(columns=['label'],axis=1)
Y = cancer_dataset_dataFrame['label']
print(X)
print(Y)
# Scalling the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Split the dataset into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)



# Create the model and preparate it
tf.random.set_seed(3)
NNModel = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20,activation='relu'),
    keras.layers.Dense(2,activation='sigmoid')
])
NNModel.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
# Import Early stopping 
Early_stopping_mintor = EarlyStopping(patience=2)
# Train the model 
result = NNModel.fit(x_train,y_train,validation_split=0.1,epochs=30,callbacks=[Early_stopping_mintor])
# Visualize the accuracy with the loss
plt.figure(figsize=(5,5))
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend(["trianing data","validation data"],loc="lower right")
# Visualize the accuracy with the loss
plt.figure(figsize=(5,5))
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["trianing data","validation data"],loc="upper right")


# Model Evaluattion
Evaluation_result = NNModel.evaluate(x_test,y_test)
print("the loss value is: ",Evaluation_result[0])
print("the accuracy value is: ",Evaluation_result[1])

# Make the model predict on testing input data by giving the prediction probability of each class(Node or group) in the output layer
predicted_test_data = NNModel.predict(x_test)
print(predicted_test_data)
predicted_test_labels = [np.argmax(prediction) for prediction in predicted_test_data]
print(predicted_test_labels)



# Build a predictive system
input_data = (13.81,23.75,91.56,597.8,0.1323,0.1768,0.1558,0.09176,0.2251,0.07421,0.5648,1.93,3.909,52.72,0.008824,0.03108,0.03112,0.01291,0.01998,     0.004506,19.2,41.85,128.5,1153,0.2226,0.5209,0.4646,0.2013,0.4432,0.1086)
# Convert input data into 1D numpy array
input_array = np.array(input_data)
# Convert 1D input array into 2D
input_2D_array = input_array.reshape(1,-1)
# Scalling the input data
input_2D_array=scaler.transform(input_2D_array)
# Make the model predict the output
if np.argmax(NNModel.predict(input_2D_array)[0])==0:
    print("the tumer is malignant")
else:
    print("the tumer is benign")



