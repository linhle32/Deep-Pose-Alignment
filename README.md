# Deep pose alignment

<b> *** Update 3/19/2021 - my paper was published in IEEE Big Data 2020 https://ieeexplore.ieee.org/abstract/document/9378321 </b>

## Summary
This deep learning architecture aligns human poses to be used in an exercise/rehabilitation assistant system. 

In short, the assistant system aims to provide users with visual feedback for their physical exercises. The feedback is generated by first extracting a user’s poses while they are performing an exercise through the video feed of the session. The extracted poses are then overlaid with the correct poses and display for the user to observe and fix their errors. 

#### An example of the full application (in development currently)

![image](https://user-images.githubusercontent.com/5643444/232255456-185a5b39-7f68-46bf-90ed-3f638eae1478.png)


This model solves the task of aligning the user’s pose with the correct pose so that they can be overlaid on each other with minimal differences, including scales, locations, and perspectives. The architecture is summarized below. The model is trained by feeding it a set of correct pose, and a set of randomly distorted poses (changes in camera locations and perspectives). The model then learns to transform the distorted poses to align with correct ones with minimal errors.

#### An example of a pose and possible changes

![image](https://user-images.githubusercontent.com/5643444/232255474-2e9d89b8-f9f5-4e60-bec6-1f2ee27c1448.png)

#### The model architecture

![image](https://user-images.githubusercontent.com/5643444/232255485-89d32b04-0088-441a-8d8a-9f5b2336701f.png)

## Libraries

- Python 3
- Tensorflow
- NumPy
- Matplotlib
- Realtime Mutli-Person Pose Estimation for pose extraction. Original repository is at https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation

## Uploaded files

- <b>utilities.py</b>: utility functions for randomizing training poses and visualizing/overlaying poses
- <b>PoseAlignModel.py</b>: class definition for the deep pose alignment model.
- <b>extract_gif_from_folder.ipynb</b>: template for using <i>Realtime Mutli-Person Pose Estimation</i> to extract pose data. Note that this notebook cannot be run as is. You need to clone the cited repository and run the code in their Python module.
- <b>deep pose alignment.ipynb</b>: training and testing a pose alignment model on sample data.
- <b>posedata.txt</b>: sample data that we collected around the Internet.

## Some examples of result

![image](https://user-images.githubusercontent.com/5643444/232255492-baebb113-2d80-42da-8466-6721efe5b163.png)

```python

```
