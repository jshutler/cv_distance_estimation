'''
Purpose: visualize data from the dataframe
'''
import os
import cv2
import pandas as pd
from math import pi



r_dirname = "../data/training/image_2/"
w_dirname = "../demo/sample_images"

df = pd.read_csv(f"{r_dirname}/../../labels.csv")
df = df[df["class"] == "Pedestrian"].head(10)
df["angle"] = df["observation angle"]


print(df.columns)

def display_img(row, attributes=["yloc", "zloc"], title="title"):
	r_fp = os.path.join(f"{r_dirname}", row['filename'].replace('.txt', '.png'))
	w_fp = os.path.join(f"{w_dirname}", row['filename'].replace('.txt', '.png'))
	print(w_fp)
	im = cv2.imread(r_fp)

	x1 = int(row['xmin'])
	y1 = int(row['ymin'])
	x2 = int(row['xmax'])
	y2 = int(row['ymax'])

	#get central line
	cv2.line(im, (int(1224/2), 0), (int(1224/2), 370), (255,255,255), 2)
	
	#get bounding box
	cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 3)

	string = f"{[row[attribute] for attribute in attributes]}" if attributes is not None else ""

	cv2.putText(im, string, (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
	cv2.imshow("detections", im)
	cv2.setWindowTitle("detections", title)
	cv2.waitKey(0)
	cv2.imwrite(w_fp, im)

if __name__ == '__main__':
	angle = (pi)
	tolerance= .1
	conds = (df["observation angle"] < angle+tolerance) & \
			(df["observation angle"] > angle-tolerance) & \
			(df["zloc"] < 30) & \
			(df["truncated"] > 0)


	disp_df = df[conds]






	

	shapes = {}
	# showing angles
	for idx, row in df.iterrows():
		# fp = os.path.join(f"{r_dirname}", row['filename'].replace('.txt', '.png'))
		# im = cv2.imread(fp)
		# shape = im.shape 
		display_img(row, None)
		# if shape in shapes.keys():
		# 	shapes[shape] += 1 
		# else:
		# 	shapes[shape] = 1

	

	# truncated_thresholds = [i/10 for i in range(1, 10)]
	# print(truncated_thresholds)
	# for threshhold in truncated_thresholds:
	#     row = truncated[truncated.truncated > threshhold].iloc[0]
	#     display_img(row, ["truncated"], title="Truncated Images")

		
		

		
		

		