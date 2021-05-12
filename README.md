# MSP-Face Database
MSP-Face is an natural audio-visual database. It is a collection of online videos, where people talk in front of the camera different kind of topics  with a diversity of emotions. We annotated the emotional content of this database. Thus, generating labeled and unlabeled parts to perform experiments. The unlabeled part of this database contains a variety of emotions and it can be used for exploring unsupervised methods. 

This is the implementation of multimodalities baseline models for the [paper](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Vidal_2020.pdf). We provide the trained baseline models for users to reproduce results in the paper.
In this repository, we share:

* Source code of baselines ('Audio_Modality','Visual_Modality', and 'AudioVisual_Modality' folders)
* Links to the online videos ('Link' folder)
* Code for getting the video segments (```download_segment_videos.py```)
* Emotional labels obtaining by crowd-sourcing annotation ('Labels' folder)
* Video segmentation times ('Doc' folder)
* Gender and Speaker identification for the video segments ('Doc' folder)
* Suggested partition of the labeled part of the database (train, test, and development sets) ('Doc' folder)

## Download MSP-Face
For downloading the videos, generating the video segments and getting the segments audio, we provide the code ```download_segment_videos.py```. The requirements for using this code are:
- python 3.6
- [pytube](https://github.com/pytube/pytube) 10.0.0
- ffmpeg
- [SoX](http://sox.sourceforge.net/Main/HomePage) 14.4.2

Installing pytube:
```
python -m pip install pytube
```

Using ```download_segment_videos.py```
```
download_segment_videos.py -l <linksFile> -s <segmentsFile> -p <saveDataPath>
```
where
- ```<linksFile>``` correspond to ```link_videos.txt```
- ```<segmentsFile>``` correspond to ```segments_data.txt```
- ```<saveDataPath>``` is the path where you want to save the database files

## Baselines
### Suggested requirements
1. Python 3.6
2. Ubuntu 18.04
3. keras version 2.2.4
4. tensorflow version 1.14.0
5. CUDA 10.0
6. opensmile-2.3.0 (users can download from the [official website](https://www.audeering.com/opensmile/))



### Audio Modality
**Step1:** We use the opensmile **IS13_ComParE.conf** feature set (apply with default configurations) for the acoustic feature extraction. The [official website](https://audeering.github.io/opensmile/about.html#capabilities) provides the detail documentation of how to extract acoustic features based on the input audio files. After feature extraction, each audio file will result in a 6373D vector representation, and then users need to save them into .mat file by the scipy.io package under whichever directory of your own local PC (i.e., YOUR_ROOT).

**Step2:** After extracted features, use the **normalization.py** in the *Audio_Modality* folder to calculate z-normalization parameters (mean and std) based on the train set. These norm-parameters will be saved in a generated *NormTerm* folder.

**Step3:** For building the audio only model, users can use the **training.py** in the *Audio_Modality* folder to train all the models by given desired parameters. The trained models and loss curves (.png) will be saved in a generated *Models* folder. After obtain the trained models, users can then run the **testing.py** file with corresponding parameter settings of the trained models to obtain prediction results based on the test set given by the corpus.

Runing args for the **training.py** and **testing.py** file are:
   * -root: your data root directory (i.e., the saved feature directory in the Step1)
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion label type ('attr' or 'class')
   * -nodes: number of nodes for Dense layers
   * -nc: number of classes for emotional classification task ('5-class' or '8-class') 
   * run in the terminal

```
python training.py -root YOUR_ROOT -ep 200 -batch 256 -emo attr -nodes 256
python testing.py -root YOUR_ROOT -ep 200 -batch 256 -emo attr -nodes 256
```
or
```
python training.py -root YOUR_ROOT -ep 200 -batch 256 -emo class -nodes 256 -nc 5-class
python testing.py -root YOUR_ROOT -ep 200 -batch 256 -emo class -nodes 256 -nc 5-class
```
NOTE: In the **testing.py** file, we also provide codes for extracting intermediate hidden outputs for AudioVisual fusion model. They are commented at the ending portion of the file.



### Visual Modality
For image based emotion recognition we first train a static emotion classification model (VGG16) on AffectNet. Using the latent feature representations we train an LSTM on the MSP_Face dataset. Then, we use the LSTM output vector to represent the whole video for Audio-visual recognition. For step-by-step of how we trained our model [click here](https://github.com/3loi/MSP_Face/tree/master/Visual_Modality)


### Audio-visual Modality
**Step1:** Use the trained audio models to extracted intermediate hidden outputs of all audios. Please refer to the ending portion (commented) of the **testing.py** file in the *Audio_Modality* folder. The extracted hidden embeddings will be saved in a generated *Fusion_Features/XXX/Audios* folder.

**Step2:** After obtain the same intermediate hidden outputs from visual models (they are saved as pickle files, for more details please refer to *Visual_Modality* folder), use **parse_video_hidden.py** in the *AudioVisual_Modality* folder to parse features and save into .mat format. The extracted hidden embeddings will be saved in a generated *Fusion_Features/XXX/Videos* folder.

**Step3:** For building the audio-visual model users can use the **fusion_model_train.py** in the *AudioVisual_Modality* folder to train all the models by given desired parameters. And test the trained model's prediction results by running **fusion_model_test.py**. 

Runing args for the **fusion_model_train.py** and **fusion_model_test.py** file are:
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion label type ('attr' or 'class')
   * -nodes: number of nodes for Dense layers
   * -nc: number of classes for emotional classification task ('5-class' or '8-class') 
   * run in the terminal

```
python fusion_model_train.py -ep 50 -batch 256 -emo attr -nodes 256
python fusion_model_test.py -ep 50 -batch 256 -emo attr -nodes 256
```
or
```
python fusion_model_train.py -ep 50 -batch 256 -emo class -nodes 256 -nc 5-class
python fusion_model_test.py -ep 50 -batch 256 -emo class -nodes 256 -nc 5-class
```



### Results for regression and classification tasks

#### Regression results
|  | Speech-only | Face-only | Audiovisual|
| --- | --- | --- | --- |
| **Aro-CCC** | 0.3794 | 0.2065 | 0.3961 |
| **Val-CCC** | 0.2924 | 0.2677 | 0.3453 |
| **Dom-CCC** | 0.3390 | 0.2085 | 0.3430 |


#### Classification results
|  | Speech-only | Face-only | Audiovisual|
| --- | --- | --- | --- |
| **5 class F1-score (macro)** | 0.2835 | 0.3027 | 0.3010 |
| **5 class F1-score (micro)** | 0.3599 | 0.3494 | 0.3641 |
| **8 class F1-score (macro)** | 0.1629 | 0.1308 | 0.1690 |
| **8 class F1-score (micro)** | 0.2637 | 0.3161 | 0.2710 |


## Reference
If you use this code or the corpus, please cite the following paper:

Andrea Vidal, Ali Salman, Wei-Cheng Lin, and Carlos Busso, "MSP-face corpus: A natural audiovisual emotional database," in ACM International Conference on Multimodal Interaction (ICMI 2020).

``` 
@InProceedings{Vidal_2020, 
  author={A. Vidal and A. Salman and W.-C. Lin and C. Busso}, 
  title={{MSP}-Face corpus: A Natural Audiovisual Emotional Database},
  booktitle={ACM International Conference on Multimodal Interaction (ICMI 2020)}, 
  volume={},
  year={2020}, 
  month={October}, 
  pages={}, 
  address =  {Utrecht, The Netherlands},
  doi={},
}
```
