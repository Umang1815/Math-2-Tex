# Math-2-Tex

This project converts Handwritten Mathematical Equations into LaTeX.

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
