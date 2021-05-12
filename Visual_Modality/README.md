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


## Training the VGG model
In this baseline we use a VGG16 model trained on AffectNet as our feature extractor. To train this model see the VGG_MODEL directory for more details. This step can be done while the previous step is running.


