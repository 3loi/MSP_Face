# Visual Modality


## preparing the Data
Once the videos have been downloaded, we follow a two step process to prepare the data.
1. Extract frames from videos
  video2jpg.py extracts the frames from video.
  ```
  python jpg2face.py -dir MSP_DIR
  ```
* -root: directory used to extract the segments and download the videos
  
3. Extract faces from frames
  jpg2face.py extracts the largest face from each frame (if found).
  ```
  python jpg2face.py -dir MSP_DIR
  ```
* -root: directory used to extract the segments and download the videos



