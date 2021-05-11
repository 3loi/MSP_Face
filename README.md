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
For downloading the videos and generating the video segments, we provide the code ```download_segment_videos.py```. The requirements for using this code are:
- python 3.6
- [pytube](https://github.com/pytube/pytube) 10.0.0
- ffmpeg

Installing pytube3:
```
python -m pip install pytube
```

Using ```download_segment_videos.py```
```
download_segment_videos.py -l <linksFile> -s <segmentsFile> -d <downloadPathOriginalVideos> -p <segmentPathDownload>
```
where
- ```<linksFile>``` correspond to ```link_videos.txt```
- ```<segmentsFile>``` correspond to ```segments_data.txt```
- ```<downloadPathOriginalVideos>``` is the path where do you want to save the original videos
- ```<segmentPathDownload>``` is the path where do you want to save the video segments extrated from the original videos

## Baselines
### Suggested requirements
1. Python 3.6
2. Ubuntu 18.04
3. keras version 2.2.4
4. tensorflow version 1.14.0
5. CUDA 10.0
6. The IS13ComParE HLDs (6373-dim acoustic features for audio modality) extracted by OpenSmile, users can download from the [official website](https://www.audeering.com/opensmile/).

### Audio Modality
For the audio only model, we put codes and trained baseline models in the *Audio_Modality* folder. The trained models are in the *Models* folder, users can directly run the **testing.py** file with corresponding parameter settings to reproduce prediction results in the paper. If users want to re-train the model from scratch by setting different parameters, different model architectures or any customize experiments, we also provide our full training procedure in the **training.py** file for your reference.

### Visual Modality



### Audio-visual Modality
We put audio-visual related codes and trained models under the *AudioVisual_Modality* folder. The hidden output features from the audio only and visual only models are saved in the *Fusion_Features* folder. We upload the trained audio-visual hidden feature fusion models in the *Fusion_Models* folder. Users can reproduce the fusion results in the paper by runing **fusion_model_test.py** directly

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
