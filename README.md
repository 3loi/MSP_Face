# MSP-Face Corpus: A Natural Audiovisual Emotional Database
This is the implementation of multimodalities baseline models for the [paper](https://ecs.utdallas.edu/research/researchlabs/msp-lab/publications/Vidal_2020.pdf). We provide the trained baseline models for users to reproduce results in the paper. 

# Suggested Environment and Requirements
1. Python 3.6
2. Ubuntu 18.04
3. keras version 2.2.4
4. tensorflow version 1.14.0
5. CUDA 10.0
6. The IS13ComParE HLDs (6373-dim acoustic features for audio modality) extracted by OpenSmile, users can download from the [official website](https://www.audeering.com/opensmile/).

# Audio Modality
For the audio only model, we put codes and trained baseline models under the *Audio_Modality* folder. The trained models are in the *Models* folder, users can directly run the **testing.py** file with corresponding parameter settings to reproduce prediction results in the paper. If users want to re-train the model from scratch by setting different parameters, different model architectures or any customize experiments, we also provide our full training procedure in the **training.py** file for your reference.

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
For runing the testing results, just change training.py into testing.py to get prediction performances based on the MSP-Face test set with correpsonding model args. 


# Vidual Modality



# AudioVidual Modality
We put audio-visual related codes and trained models under the *AudioVisual_Modality* folder. The hidden output features from the audio only and visual only models are saved in the *Fusion_Features* folder. We upload the trained audio-visual hidden feature fusion models in the *Fusion_Models* folder. Users can reproduce the fusion results in the paper by runing **fusion_model_test.py** directly.

Runing args for the **fusion_model_train.py** and **fusion_model_test.py** file are:
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion label type ('attr' or 'class')
   * -nodes: number of nodes for Dense layers
   * -nc: number of classes for emotional classification task ('5-class' or '8-class') 
   * run in the terminal

```
python fusion_model_train.py -ep 50 -batch 256 -emo attr -nodes 256
```
or
```
python fusion_model_train.py -ep 50 -batch 256 -emo class -nodes 256 -nc 5-class
```
For runing the testing results, just change fusion_model_train.py into fusion_model_test.py to get fusion performances based on the MSP-Face test set with correpsonding model args. 


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
