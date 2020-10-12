# MSP-Face Corpus: A Natural Audiovisual Emotional Database
This is the implementation of multimodalities baseline models for the [paper](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Vidal_2020.pdf). We provide the trained baseline models for users to reproduce results in the paper. 

# Suggested Environment and Requirements
1. Python 3.6
2. Ubuntu 18.04
3. keras version 2.2.4
4. tensorflow version 1.14.0
5. CUDA 10.0
6. The MSP-Face corpus (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Face.html))
7. The IS13ComParE HLDs (6373-dim acoustic features for audio modality) extracted by OpenSmile, users can download from the [official website](https://www.audeering.com/opensmile/).
8. ...

# Audio Modality
For the audio only model, we put codes and trained baseline models under the *Audio_Modality* folder. The trained models are in the *Models* folder, users can directly run the **testing.py** file with corresponding parameter settings to reproduce prediction results in the paper. If users want to re-train the model from scratch by setting different parameters, different model architectures or any customize experiments, we also provide our full training procedure in the **training.py** file for your reference.

# Vidual Modality



# AudioVidual Modality
We put audio-visual related codes and trained models under the *AudioVisual_Modality* folder. The hidden output features from the audio only and visual only models are saved in the *Fusion_Features* folder. We upload the trained audio-visual hidden feature fusion models in the *Fusion_Models* folder. Users can reproduce the fusion results in the paper by runing **fusion_model_test.py** directly.


# Reference
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
