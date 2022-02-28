import pandas as pd 
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import os 
import cv2
from sklearn.model_selection import train_test_split
from math import sin, cos

dirname = "../data/training/subimages/"


def float_range(start, stop, step):
	while start < stop:
		yield float(start)
		start += step

def display_img(row, attributes=["yloc", "zloc"], title="title"):
	fp = os.path.join(f"{dirname}", row['filename'].replace('.txt', '.png'))
	img = cv2.imread(fp)

	x1 = int(row['xmin'])
	y1 = int(row['ymin'])
	x2 = int(row['xmax'])
	y2 = int(row['ymax'])

	#get central line
	cv2.line(img, (int(1224/2), 0), (int(1224/2), 370), (255,255,255), 2)
	
	#get bounding box
	cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

	string = f"{[row[attribute] for attribute in attributes]}"

	cv2.putText(img, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	cv2.imshow("detections", img)
	cv2.setWindowTitle("detections", title)
	cv2.waitKey(0)
	# print(im.shape)
	# print(x1, x2, y1, y2)
	cv2.imshow("sub_image", img[y1:y2, x1:x2])
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

def pad_image(img, max_shape=(376, 1242)):

	# fp = os.path.join(f"{dirname}", row['filename'].replace('.txt', '.png'))
	# img = cv2.imread(fp)

	max_height = max_shape[0]
	max_width = max_shape[1]
	padded_img = cv2.copyMakeBorder(img, max_height - img.shape[0], 0, max_width - img.shape[1], 0, cv2.BORDER_CONSTANT, value=0)

	scale_percent = .25
	new_shape = (int(padded_img.shape[1] * scale_percent), int(padded_img.shape[0] * scale_percent))
	
	resized_img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)


	return resized_img

def save_bb_image(row, out_file):
	#get file
	fp = os.path.join(f"{dirname}", row['filename'].replace('.txt', '.png'))
	img = cv2.imread(fp)

	#get bounding box
	x1 = int(row['xmin'])
	y1 = int(row['ymin'])
	x2 = int(row['xmax'])
	y2 = int(row['ymax'])
	
	cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

	padded_img = pad_image(img)

	scale_percent = .25


	new_shape = (int(padded_img.shape[1] * scale_percent), int(padded_img.shape[0] * scale_percent))
	
	
	resized_img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)


	cv2.imwrite(outfile, resized_img)



def get_padded_subimage(row, max_shape=(376, 1242)):
	fp = os.path.join(f"{dirname}", row['filename'].replace('.txt', '.png'))
	img = cv2.imread(fp)

	#get bounding box
	x1 = int(row['xmin'])
	y1 = int(row['ymin'])
	x2 = int(row['xmax'])
	y2 = int(row['ymax'])

	print(img)
	mask = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

	subimage = img[y1: y2, x1:x2]

	max_height = max_shape[0]
	max_width = max_shape[1]
	padded_img = cv2.copyMakeBorder(subimage, max_height - y2, y1, x1, max_width - x2, cv2.BORDER_CONSTANT, value=0)


	# cv2.imshow("image", img)
	# cv2.waitKey(0)
	# cv2.imshow("subimage", padded_img)
	# cv2.waitKey(0)
	return padded_img
	


df = pd.read_csv("../data/train.csv")
df["dist_feet"] = df["zloc"].apply(lambda x: x * 3.28084)

df["pixel_width"] = df.xmax - df.xmin
df["pixel_height"] = df.ymax - df.ymin

#get our discrete angles
df["angle"] = df["observation angle"].apply(lambda angle: angle + 2*pi if angle < 0 else angle)

df["angle_discrete"] = df["angle"].apply(lambda angle: make_discrete_angle(angle))

df["cos_angle"] = df["angle"].apply(lambda x: cos(x))
df["sin_angle"] = df["angle"].apply(lambda x: sin(x))

length_max = 0
width_max = 0
padding_shape = (374, 510)

dirname = "/home/jack/Desktop/code/projects/deloitte/cv_distance_estimation/data/training/subimages_padded_no_position/"

shapes = []
for idx, row in df.iterrows():
	print(idx)
	fp = f"{dirname}{row['filename_bb'].replace('.txt', '.png')}"
	print(fp)
	im = cv2.imread(fp)
	if im is not None:
		shapes.append(im.shape)
	# padded_img = pad_image(im, padding_shape)


print(set(shapes))

	

	# cv2.imwrite(f"../data/training/subimages_padded_no_position/{idx}.png", padded_img)
	# print(idx)
	# outfile = f"data/training/image_2_bounding/{idx}.png"
	# save_bb_image(row, outfile)
print((length_max, width_max))

# df_train, df_test =  train_test_split(df)

# # #save the bois
# df_train.to_csv("data/train.csv", index=False)
# df_test.to_csv("data/test.csv", index=False)

# df.to_csv("data/labels.csv", index=False)

# # print(df.columns)
