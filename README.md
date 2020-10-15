# MSP-Face Database
MSP-Face is an natural audio-visual database. It is a collection of online videos, where people talk in front of the camera different kind of topics  with a diversity of emotions. We annotated the emotional content of this database. Thus, generating labeled and unlabeled parts to perform experiments. The unlabeled part of this database contains a variety of emotions and it can be used for exploring unsupervised methods. In this repository, we share:

* Source code of baselines ('codes' folder)
* Links to the online videos ('Link' folder)
* Code for getting the video segments (```download_segment_videos.py```)
* Emotional labels obtaining by crowd-sourcing annotation ('Labels' folder)
* Video segmentation times ('Doc' folder)
* Gender and Speaker identification for the video segments ('Doc' folder)
* Suggested partition of the labeled part of the database (train, test, and development sets) ('Doc' folder)

## Download MSP-Face
For downloading the videos and generating the video segments, we provide the code ```download_segment_videos.py```. The requirements for using this code are:
- python 3.6
- [pytube3](https://github.com/get-pytube/pytube3)
- ffmpeg

Installing pytube3:
```
pip install pytube3 --upgrade
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

## Emotional Labels
The labels are in JSON format.
* ```labels_concensus.json``` includes emotion labels(class, three attributes), gender, speaker and split set information.
* ```labels_detailed.json``` includes individual annotations provided by each worker to each sentence

The "**labels_concensus**" file is organized as follows:

```
{
    "MSP-FACE_0253_0006.mp4": [
        {
            "EmoClass_Major": "H", #Consensus categorical class
            "EmoAct": "5.428571428571429", #Activation attribute value
            "EmoDom": "5.0", #Dominance attribute value
            "EmoVal": "4.857142857142857", #Valence attribute value
            "SpkrID": "27",
            "Gender": "Female",
            "Split_Set": "Train"
        }
    ],...
}
```
The "**labels_details**" file provides all the annotations per video. The file is organized as follows:
```
{
    "MSP-FACE_0253_0006.mp4": {
        "Workers": [
            {
                "Name": "WORKER0000013",
                "EmoClass_Major": "Happy", #Categorical class
                "EmoClass_Second": "Happy,Concerned,Excited", #Secondary categorical emotions selected
                "EmoAct": "6", #Activation attribute value
                "EmoDom": "5", #Dominance attribute value
                "EmoVal": "5" #Valence attribute value
            },
            {
                "Name": "WORKER0000001",
                "EmoClass_Major": "Happy",
                "EmoClass_Second": "Happy",
                "EmoAct": "6",
                "EmoDom": "6",
                "EmoVal": "6"
            },...
	]
...
}
```
## Speaker information
We have manually annotated the speaker identity of 27,325 sentences corresponding to 491 speakers. The file **speaker_id_segments.txt** provides the details. The file **speaker_id_gender.txt** provides the gender of the speakers. Both files are in "**Doc**" folder.

## Baselines
Tables below shows the baselines for regression and classification tasks. We provide the source code for replicating the results from our paper.

### Regression results
|  | Speech-only | Face-only | Audiovisual|
| --- | --- | --- | --- |
| **Aro-CCC** | 0.3794 | 0.2065 | 0.3961 |
| **Val-CCC** | 0.2924 | 0.2677 | 0.3453 |
| **Dom-CCC** | 0.3390 | 0.2085 | 0.3430 |

### Classification results
|  | Speech-only | Face-only | Audiovisual|
| --- | --- | --- | --- |
| **5 class F1-score (macro)** | 0.2835 | 0.3027 | 0.3010 |
| **5 class F1-score (micro)** | 0.3599 | 0.3494 | 0.3641 |
| **8 class F1-score (macro)** | 0.1629 | 0.1308 | 0.1690 |
| **8 class F1-score (micro)** | 0.2637 | 0.3161 | 0.2710 |

## Reference
If you use this database, please cite the following paper:
"MSP-Face Corpus: A Natural Audiovisual Emotional Database," International Conference on Multimodal Interaction (ICMI), 2020.
```
@InProceedings{xXx,
  title={XXX},
  author={XXX},
  booktitle={XXX},
  year={2020},
} 
```
