# GZTAN Music Classification




Setup
-----
Prerequisites: brew ([brew.sh](brew.sh) for installation instructions)

```
brew install pyenv pyenv-virtualenv graphviz direnv
```

Direnv is used to create, activate, and deactivate the virtual environment for this project.  Upon first use and edit, you must allow direnv to make changes to your system.

```
direnv allow
```

Once the virtual environment is activated, install the python requirements via:

```
pip install -r requirements.txt
```

Data was downloaded from Kaggle ([link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)) due to tensorflow datasets timing out at time of development.  Ensure that the location of the dataset corresponds to the arguments passed into classify/prepare_dataset.py

Running the code
-----
The data is prepared using `classify/prepare_dataset.py`, you can use the default arguments or override them as needed.  The network is build, trained, and tested using `classify/train_and_predict.py`.

Results
-----
This model achieves a 63.33% accuracy on the test set with the best model (based on validation accuracy) after 200 epochs of training

![Accuracy](results/accuracy-gztan-dataset-2023-11-17-22.00.png)
![Loss](results/loss-gztan-dataset-2023-11-17-22.00.png)
