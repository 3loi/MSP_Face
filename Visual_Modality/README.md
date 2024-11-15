# Visual Modality


## Preparing the Data
Once the videos have been downloaded, we follow a two step process to prepare the data.
1. Extract frames from videos.
  video2jpg.py extracts the frames from video.
  ```
  python jpg2face.py -dir MSP_DIR
  ```
* -dir: directory used to extract the segments and download the videos
  
2. Extract faces from frames.
  jpg2face.py extracts the largest face from each frame (if found).
  ```
  python jpg2face.py -dir MSP_DIR
  ```
* -dir: directory used to extract the segments and download the videos


One the data has been pre-processed and we have trained our [VGG models](https://github.com/3loi/MSP_Face/tree/master/Visual_Modality/VGG_Model) we can start training our LSTM model. Before that please make sure the config file (config_file.py) is configured correctly.

3. Extract the latext feature from VGG model (uses config_file.py) on the MSP_Face Dataset
  ```
  python vgg_feature_extract.py
  ```
  
4. Train the model and save the video representation as a pickle file (uses config_file.py)
  ```
  python train_LSTM.py
  ```
