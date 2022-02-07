import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import cv2
# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

import time

def get_encoding(series):
	#getting our classes
	array = series.to_numpy().reshape(-1,1)
	encoder = OneHotEncoder().fit(array)
	return encoder.transform(array).toarray()

def get_images(df, img_dir):

	images = []

	for idx, row in df.iterrows():
		file = row["filename"].replace(".txt", ".png")
		x1, y1, x2, y2 = (int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))


		im = cv2.imread(f"{img_dir}/{file}")
		cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)
		resized = cv2.resize(im, (1224, 370))

		images.append(resized)

	return np.array(images)

def train(df_train, df_test, model_name, epochs=5000):
	X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
	y_train = df_train[['dist_feet']].values

	X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
	y_test = df_test[['dist_feet']].values

	# standardized data
	scalar = StandardScaler()
	X_train = scalar.fit_transform(X_train)
	y_train = scalar.fit_transform(y_train)

	# ----------- create model ----------- #
	model = Sequential()
	model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

	model.compile(loss='mean_squared_error', optimizer='adam')

	# ----------- define callbacks ----------- #
	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
									   verbose=1, epsilon=1e-4, mode='min')
	
	tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))

	# ----------- start training ----------- #
	history = model.fit(X_train, y_train,
	 							 validation_split=0.1, epochs=epochs, batch_size=2048,
	 							 callbacks=[tensorboard], verbose=1)

	# ----------- save model and weights ----------- #
	model_json = model.to_json()
	with open("models/{}.json".format(modelname), "w") as json_file:
	    json_file.write(model_json)

	model.save_weights("models/{}.h5".format(modelname))
	print("Saved model to disk")







if '__main__' == __name__:
	
	img_dir = "data/training/image_2"
	
	n = 5
	
	df = pd.read_csv("labels.csv")

	model_name = "bounding_box_only"

	df["dist_feet"] = df["zloc"].apply(lambda x: x * 3.28084)


	df_train, df_test =  train_test_split(df)

	df_train.to_csv("train.csv")
	df_test.to_csv("test.csv")



	input_features = {}
	output_features = {}

	
	input_features["bounding_boxes"] = df_train[["xmin", "xmax", "ymin", "ymax"]].to_numpy()
	input_features["class"] = get_encoding(df_train["class"])
	input_features["images"] = get_images(df_train, img_dir)

	print(input_features.images.head())
	
	# output_features["angle_discrete"] = get_encoding(df_train["angle_discrete"])
	# output_features["distance"] = df_train.zloc.to_numpy().reshape(-1,1)
	

	# train(df_train, df_test, model_name)
