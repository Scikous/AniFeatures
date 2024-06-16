# AniFeatures
Based on DeepDanbooru by KichangKim, but semi-simplified and works using PyTorch over TensorFlow among some other updates.

> :information_source: Check out the acknowledgements section for particular references 

Using DeepDanbooru to train a custom model is unfortunately rather difficult to setup and get working if at all on the GPU (I tried and decided it was faster to just rebuild everything necessary). TensorFlow from 2.10 onwards no longer supports running on the GPU on Windows according to [Kiran_Sai_Ramineni](#acknowledgements), through my own experiments I've also confirmed this to be the most likely case. The only theoretical way to work with Windows would've been to downgrade TensorFlow and install the supported CUDA-toolkit and CUDNN, this would've been too much of a hassle and I have no guarantee that they would work with my hardware, so I didn't.

# Features
* Download images from Danbooru alongside their tags
* Train a custom model for detecting tags in an image
* Download all tags from Danbooru


# Todo
* [ ] tqdm progress bar to know how training is progressing
* [ ] a full release with a pre-trained model and downloader.exe
* [X] save best model

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
python -m DanbooruDownloader dump MyDataset --username <Danbooru username> --api-key <Danbooru api key> 
```

If you don't already have an account or the API key, this is rather straight-forward. I'll skip over the account creation, should be simple enough. For the API key click on **My Account** (top left), and then at the bottom is **API Key**, clicking on **view** will bring you to the API key page where all of your API keys reside, create a new key with **Add** (top right). You can, but don't need to fill in or select anything, just pressing **Create** is enough. Back on the API keys page, the **Key** column will contain the key needed for the downloader.

From everything that is downloaded, only the **images** and the **.sqlite** database are of importance.

# Data preprocessing
The downloaded images and the corresponding tags will now need to be preprocessed a bit. Using `data_preprocessor.py`, first, all of the images are tested to find any corrupted images, which are then deleted. Next, all of the **image names** and their corresponding **extensions** from the **.sqlite** database are collected and concatenated, and each image's corresponding list of tags are also collected, all of this is written to a single .csv file. The images will lastly be moved from the subdirectories into a single main directory.

:warning: **csv_to_csv** function should always be run after **db_to_csv**
It is also possible to use the **csv_to_csv** function to transfer custom images over to the final `metadata.csv`.

# Model Training
The model training is simple enough, just running the `train.py` file will be enough, provided the data exists and is in the expected format (preprocessing step should handle this). By default the GPU is used (specifically GPU 0, so be sure to check that's the one you want to use), but if it can't be, then the trainer will use the CPU.

The tags are binarizied for faster computing. The DataLoader alongside Dataset modules from torch manage the dataset and splits the data into batches for training.

The tags.txt file that is created, is created and used for evaluation (easier and less redundant than reading the entire .csv file again).

# Unknown
The `tags.py` (based on [bem13's](#acknowledgements) script) retrieves (or can at least) all of the possible tags from Danbooru. Functionally it currently has no purpose as the evaluation requires for the model and list of tags to have the same amount. But maybe it will have a use case eventually (doubt it). So, think of it as an early legacy feature.


# Acknowledgements
* [bem13 - danbooru_tag_count_scraper](https://gist.github.com/bem13/596ec5f341aaaefbabcbf1468d7852d5)
* [DeepDanbooru](https://github.com/KichangKim/DeepDanbooru)
* [DanbooruDownloader](https://github.com/KichangKim/DanbooruDownloader)
* [thisismycontributionaccount](https://github.com/KichangKim/DanbooruDownloader/pull/16)
* [Kiran_Sai_Ramineni](https://discuss.tensorflow.org/t/tensorflow-cannot-detect-gpu/23006)