# CV Distance Estimation Project
The goal of this project is to train a CV model that can take an image of a target, with the coordinate bounding box of that target, and provide the distance the target is from the camera. 

## Why this project
A long time goal of mine is to escape the 1st person existance I was born into, and live in the 3rd person, the same way (I can control Mario in Super Mario 64)[https://www.youtube.com/watch?v=O9GH3Pp9qGA]. That is when I had the idea to make this dream come true.
1. Buy an (FPV drone)[https://www.youtube.com/watch?v=0Wd7ZhGSkRo]
2. Write some software that can take the video feed from the drone and give the X, Y, Z coordinates of some target (myself)
3. Tell drone to follow a target (me) within some distance automatically.
4. My dreams of being Mario have come true.

The plan is pretty simple, but the devil is in the details. AI can already provide an X and Y coordinate in the form of a bounding box on a target, but getting Z (depth from camera to target) is not trivial using purely images. So all that is left is to train an AI model to calculate the depth of the target. 

## The Kitti Self Driving Dataset
Thankfully, the self-driving community has already created a dataset to attempt to accomplish this task in the (Kitti Dataset)[https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction]. This provides me with the key data that I need.
1. Images
2. CSV with metadata about targets in the images:
  a. X, Y coordinates for a bounding box
  b. Z Depth Coordinates (What I want to predict)
  c. Angle of center of object from camera
  d. What kind of object am I looking at (Car, pedestrain, cyclist, etc)
  e. Truncation - How much the image is removed from the image boundries
  f. Occlusion - How much is the target being obstructed from the camera's view by other objects

You can see a EDA deep dive into the data in the 'Kitti EDA.ipynb'

## Training
I trained a few different models to see how they would all function using different input~output pairs.

_only predicting the depth_
1. bounding box (no image) ~ depth
2. bounding box + image ~ depth
3. bounding box + image + object class
4. bounding box + subimage (pixels outside of bounding box are zeroed out) ~ depth
_predicting the depth and the angle_
1. bounding box (no image) ~ depth + angle
2. bounding box + image ~ depth + angle
3. bounding box + image + object class + angle
4. bounding box + subimage (pixels outside of bounding box are zeroed out) ~ depth + angle

My code to train these I then proceeded to train my model in the 'train_image_models.ipynb'
