# Machine Learning Engineer Nanodegree
## Capstone Proposal and Capstone Project
## Project: Using the YOLO (You Look Only Once) Algorithm for People Detection

###Overview

Accurate and fast object detection is a domain which is continually being improved due to the many machine learning applications where it is applied. These areas encapsulate autonomous driving, video surveillance, and much more. This machine learning capstone project seeks to use object detection to identify people in a video stream. The goal is to develop an understanding of the current work being done in the field and implement the YOLO architecture on a specific problem, people detection. YOLO was documented with a frames per second (FPS) speed of 155 and mean average precision (mAP) of 52.7%. The goal of this project is to improve accuracy when implementing on one class, people, and show that the model can be retrained for a variety of different classes and / or objects. 

###YOLO Architecture

YOLO is a popular object detection algorithm allowing for real time object detection, but at the expense of accuracy. The model divides the image into a SxS grid and identifies B bounding boxes for each cell of the grid, with each box having a confidence value. Each cell of the grid is also assigned a class probability. Non-maximum suppression is then used to combine multiple bounding boxes which are identifying the same object. See image below for a visual of how the model operates.  

<p align="center">
  <img src="report-images/yolo.png" width="350">
</p>