# Emotional Labels
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
# Speaker information
We have manually annotated the speaker identity of 27,325 sentences corresponding to 491 speakers. The file **speaker_id_segments.txt** provides the details. The file **speaker_id_gender.txt** provides the gender of the speakers. Both files are in "**Doc**" folder.
