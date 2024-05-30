# AniFeatures
Based on DeepDanbooru by KichangKim, but semi-simplified and works using PyTorch over TensorFlow among some other updates.

> :information_source: Check out the acknowledgements section for particular references 

Using DeepDanbooru to train a custom model is unfortunately rather difficult to setup and get working if at all on the GPU (I tried and decided it was faster to just rebuild everything necessary). TensorFlow from 2.10 onwards no longer supports running on the GPU on Windows according to [Kiran_Sai_Ramineni](#acknowledgements), through my own experiments I've also confirmed this to be the most likely case. The only theoretical way to work with Windows would've been to downgrade TensorFlow and install the supported CUDA-toolkit and CUDNN, this would've been too much of a hassle and I have no guarantee that they would work with my hardware, so I didn't.


# Features
* Fine-tune an LLM
* Train and use a custom voice model for TTS (see acknowledgements)
* Speak to your LLM using STT (see acknowledgements)


## Table of Contents

* [Vtuber-AI](#vtuber-ai)
    * [Features](#features)
    * [Todo List](#todo-list)
* [Setup](#setup)
    * [Virtual Environments](#virtual-environments)
* [Large Language Model (LLM)](#large-language-model-llm)
    * [Prompt Style](#prompt-style)
    * [Dataset preparation](#dataset-preparation)
    * [Training (Fine-tuning)](#training-fine-tuning)
    * [Inference](#inference)
* [Voice Model](#voice-model)
    * [Training](#training)
        * [Official Guide](#official-guide)
        * [Unofficial Guide (Windows Only)](#unofficial-guide-windows-only)
            * [Dataset Creation](#dataset-creation)
            * [Preprocessing Audio](#preprocessing-audio)
            * [Audio Labelling](#audio-labelling)
    * [Training](#training-1)
    * [Inference](#inference-1)
* [Acknowledgements](#acknowledgements)

# Setup
This is developed and tested on Python 3.11.8.

In the root directory, install everything in the requirements.txt. All of the necessary packages are included in this file, no need to install anything anywhere else. (I highly suggest setting up a virtual environment up first).
```
pip install -r requirements.txt
```
## Virtual Environments
The virtual environment simply helps to avoid package conflicts. Do note that this will take more space in the storage as each environment is its own.

:information_source: Note that this is for CMD

Create env (the last `venv` is the folder name/path where the venv will be created):
```
 python -m venv venv
```

Activate env:
```
venv\Scripts\activate
```

Deactivate env:
```
 deactivate
```
Then just delete the venv folder


# DanbooruDownloader
:warning: There's no guarantee this will work long term, and the download method creats a little bit of excess in our use case. But, I have no short term desire to create something better as this works.

The base DanbooruDownloader is broken, but thanks to the solutions provide by [thisismycontributionaccount](#acknowledgements) building and using the downloader is possible. This fix has already been implemented in this version, so all one needs to do is build it. A pre-built release is WIP.

The overall usage is simple enough, in the directory in which the `DanbooruDownloader.exe` resides in run the following command:

:information_source: username is not to be confused with the User ID
```
python -m DanbooruDownloader dump MyDataset --username <danbooru username> --api-key <danbooru api key> 
```

If you don't already have an account or the API key, this is rather straight-forward. I'll skip over the account creation, should be simple enough. For the API key click on **My Account** (top left), and then at the bottom is **API Key**, clicking on **view** will bring you to the API key page where all of your API keys reside, create a new key with **Add** (top right). You can, but don't need to fill in or select anything, just pressing **Create** is enough. Back on the API keys page, the **Key** column will contain the key needed for the downloader.

From everything that is downloaded, only the **images** and the **.sqlite** database are of importance.

# Data preprocessing
The downloaded images and the corresponding tags will now need to be preprocessed a bit. `data_preprocessor.py` will get all of the **image names** and their corresponding **extensions** from the **.sqlite** database and concatenate them, and gets each image's corresponding list of tags, all of this is written to a single .csv file.

The images will also be moved from the subdirectories into a single main directory.




# Acknowledgements
* [bem13 - danbooru_tag_count_scraper](https://gist.github.com/bem13/596ec5f341aaaefbabcbf1468d7852d5)
* [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)
* [DanbooruDownloader](https://github.com/KichangKim/DanbooruDownloader)
* [thisismycontributionaccount](https://github.com/KichangKim/DanbooruDownloader/pull/16)
* [Kiran_Sai_Ramineni](https://discuss.tensorflow.org/t/tensorflow-cannot-detect-gpu/23006)