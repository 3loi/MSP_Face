# VGG model

To train the model we first create a .h5 file of the [AffectNet corpus](http://mohammadmahoor.com/affectnet/). Once you have the AffectNet corpus modify the config_file.py file. 

* IMAGES_PATH: path to the root of the AffectNet corpus (i.e., /Desktop/cropped_Annotated)
* NUM_CLASSES: 8, 5, or None (VAD)
* OUTPUT_PATH: output path for the trained model


Once that is complete run build_affectnet.py followed by train_vgg.py

