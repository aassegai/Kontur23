# Kontur23
This repo represents my solution of the Kontur 2023 internship test task. The task was to find substrings in the document by requested label. In simple words, it is a classic NER problem statement, but with custom labels. 

As a solution I propose a Transformer-based neural network trained with SpaCy library. 

# Contents
## [data](/data)
Data folder consists of train and test dataset given by the organizers.
## [src](/src)
Src folder contains subfolders with modules.
### [src/preprocessing](/src/preprocessing)
Contains python modules that are concerned on preprocessing raw data for using it in model training:
[data_preprocessor.py](/src/preprocessing/data_preprocessor.py) is used for transforming raw data with different text preprocessing options.
[prepare_bin.py](/src/preprocessing/prepare_bin.py) module is needed to make doc_bin files from the data. These bins are used to put the data into a spacy model (by analogy to torch Dataset class).
### [src/model]
[spacy_cfg.cfg](/src/model/spacy_cfg.cfg) is a spacy model user configuration file. It is used to put the model and training parameters in the trainer.
[predict.py](/src/model/predict.py) module is used to retrieve predicted entities for given text.
