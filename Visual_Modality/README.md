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




-------------------------------------
Runing args for the **training.py** and **testing.py** file are:
   * -root: your data root directory
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion label type ('attr' or 'class')
   * -nodes: number of nodes for Dense layers
   * -nc: number of classes for emotional classification task ('5-class' or '8-class') 
   * run in the terminal

```
python training.py -root YOUR_ROOT -ep 200 -batch 256 -emo attr -nodes 256
```
or
```
python training.py -root YOUR_ROOT -ep 200 -batch 256 -emo class -nodes 256 -nc 5-class
```
