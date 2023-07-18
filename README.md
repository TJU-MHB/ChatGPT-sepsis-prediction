# ChatGPT-sepsis-prediction

This repository contains the data and code for our experiments.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

###Experimental Environment
Environment| Description|
---|---------|
Language| Python3.7|
frame| Pytorch1.6|
IDE|Pycharm and Colab|
Equipment| CPU and GPU|

### Prerequisites

for the experiments a GPU-based Pytorch environment is assumed. For packages see requirements.txt and run:

```
$ pip install -r requirements.txt
```
Call the ChatGPT API to process text data

```
$ python3 -m API/api-scrip.py
```

Run the entire program quickly

```
$ python3 -m BILSTM/train.py
```

In BILSTM/data folder has a demo set of our final data

##Requirements for MIMIC:
  a) Requesting Access to MIMIC (publicly available, however with permission procedure)
      https://mimic.physionet.org/gettingstarted/access/
  b) Downloading and installing the MIMIC database according to documentation: 
      https://mimic.physionet.org/gettingstarted/dbsetup/  
      unix/max: https://mimic.physionet.org/tutorials/install-mimic-locally-ubuntu/  
      windows: https://mimic.physionet.org/tutorials/install-mimic-locally-windows/ 

