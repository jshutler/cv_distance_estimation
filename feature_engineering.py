import pandas as pd 
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import os 
import cv2
from sklearn.model_selection import train_test_split

dirname = "data/training/image_2/"


def float_range(start, stop, step):
	while start < stop:
		yield float(start)
		start += step

def display_img(row, attributes=["yloc", "zloc"], title="title"):
	fp = os.path.join(f"{dirname}", row['filename'].replace('.txt', '.png'))
	im = cv2.imread(fp)

	x1 = int(row['xmin'])
	y1 = int(row['ymin'])
	x2 = int(row['xmax'])
	y2 = int(row['ymax'])

	#get central line
	cv2.line(im, (int(1224/2), 0), (int(1224/2), 370), (255,255,255), 2)
	
	#get bounding box
	cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)

	string = f"{[row[attribute] for attribute in attributes]}"

	cv2.putText(im, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	cv2.imshow("detections", im)
	cv2.setWindowTitle("detections", title)
	cv2.waitKey(0)
	# print(im.shape)
	# print(x1, x2, y1, y2)
	cv2.imshow("sub_image", im[y1:y2, x1:x2])
	cv2.setWindowTitle("detections", title)
	cv2.waitKey(0)


def make_discrete_angle(angle):

	bins = [(angle, (angle + pi/4)) for angle in float_range(pi/8, (15*pi / 8), pi/4)]

	directions = ["down-right", "down", "down-left", "left", "up-left", "up", "up-right"]

	for i, b in enumerate(bins):
		if b[0] < angle < b[1]:
			return directions[i]

	#then we've run the circle and need the last one, 
	#this is me not wanting to deal with modular arithmatic :)
	return "right"







df = pd.read_csv("labels.csv")

df["dist_feet"] = df["zloc"].apply(lambda x: x * 3.28084)

df["pixel_width"] = df.xmax - df.xmin
df["pixel_height"] = df.ymax - df.ymin

#get our discrete angles
df["angle"] = df["observation angle"].apply(lambda angle: angle + 2*pi if angle < 0 else angle)

df["angle_discrete"] = df["angle"].apply(lambda angle: make_discrete_angle(angle))


df_train, df_test =  train_test_split(df)

#save the bois
df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)

df.to_csv("labels.csv", index=False)

print(df.columns)
