# Math-2-Tex

In today's scenario LaTex is used in almost all the field of study from writing a small lab report to publishing large research paper. LaTex is similar to code so many people sometimes find it tough to write in LaTex especially when it comes to writing a complex mathematical equation. Keeping this in mind we have developed a model that would create a LaTex code of any mathematical equation given it's handwritten image.

#  Methodology

- Preprocessed the images and their LaTeX labels, from corresponding inkml files
- Trained an Encoder-Decoder model with EfficientNet-B6 as encoder and LSTM Cell with Attention as decoder. 
- Achieved a Top-5 accuracy of 95% and Top-1 accuracy of 87% and deployed the model using Flask.

# User Guide

Steps to run this on your local computer:

- Clone this repository
```
git clone https://github.com/Umang1815/Math-2-Tex
```

- Make a new virtual environment in python in the folder in which this repository is saved and Active the Environment.
```
pip install virtualenv
python -m venv <myenvname> 
path\to\venv\Scripts\Activate.ps1  (Run this line with your path to activate the virtual environment)
```
- Download the requirements of the environment using 
```
pip install -r requirements.txt
```
- Download the model weights and place them into "model_data" folder. You can get the model weights [here](https://drive.google.com/drive/folders/1t7p0JlcxDTcNWR1Gu4SZnOo9RNmXEEBI?usp=sharing).

- Now, run app.py file to open the website
```
flask run
```
# Website Screenshots

