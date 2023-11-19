GZTAN Music Classification

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
