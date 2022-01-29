'''
Purpose: visualize data from the dataframe
'''
import os
import cv2
import pandas as pd
from math import pi
df = pd.read_csv("labels.csv")

df["angle"] = df["observation angle"]


dirname = "data/training/image_2/"
print(df.columns)

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


if __name__ == '__main__':
	angle = (pi)
	tolerance= .1
	print(angle)
	conds = (df["observation angle"] < angle+tolerance) & \
			(df["observation angle"] > angle-tolerance) & \
			(df["zloc"] < 30) & \
			(df.truncated > 0)


	disp_df = df[conds]
	
	print(disp_df.shape)

	#showing angles
	# for idx, row in disp_df.head().iterrows():
	# 	display_img(row, ["angle"], title)



	truncated_thresholds = [i/10 for i in range(1, 10)]
	print(truncated_thresholds)
	for threshhold in truncated_thresholds:
	    row = truncated[truncated.truncated > threshhold].iloc[0]
	    display_img(row, ["truncated"], title="Truncated Images")

		
		

		
		

		