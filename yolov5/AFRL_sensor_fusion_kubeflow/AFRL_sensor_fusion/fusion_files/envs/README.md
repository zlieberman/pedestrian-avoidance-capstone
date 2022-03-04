# Setups
It is highly recommend to use virtualenv for different purpose. All the following methods are tested on the SICA server. The requirements files are for pip only. If you are using conda to setup the enviroment, you might need to change the name of certain package.

## Setup virtualenvwrapper (Optional)
### Setup
You can setup virtual enviroment anyway you want. The following is a tutorial to setup virtualenvwrapper which will make your life a lot easier with virtual enviroment. 
- install pip if you are not already

  ```sudo apt-get install python3-pip```

- install `virtualenvwrapper`

  Install locally (**you should do this if you are on the server**)

  ```pip3 install --user virtualenvwrapper```

  or 

  ```sudo pip3 install virtualenvwrapper```
  
- Shell setup
  
  **If you are on the server** add the following line to the `.bashrc`
  ```
  export WORKON_HOME=$HOME/envs
  export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
  source $HOME/.local/bin/virtualenvwrapper.sh
  ```
  
  **If you install it on your system** add the following line to the `.bashrc`
  ```
  export WORKON_HOME=$HOME/envs
  export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
  source /usr/local/bin/virtualenvwrapper.sh
  ```
  
  Note: WORKON_HOME is where they store the enviroment (you can name whatever you want), VIRTUALENVWRAPPER_PYTHON is the default python to use when create the virtual enviroment
  
  Source the `.bashrc` of open a new terminal
  
  ```source ~/.bashrc```
  
### How to use
Create new virtual enviroment name `(ENV_NAME)`

```mkvirtualenv ENV_NAME```

Create new virtual enviroment using certain version of python and requirements files

```mkvirtualenv ENV_NAME -r /PATH/TO/REQUIREMENTS -p /PATH/TO/PYTHON```

```mkvirtualenv yolo-a-train -r requirements_YOLOv3_archive.txt -p /usr/bin/python3.8```

Start the virtual enviroment

```workon ENV_NAME```

Stop the virtual enviroment

```deactivate```

## Regular enviroment
This is for running the sensor fusion

**pip**
```python3 -m pip install -r requirements.txt```

**mkvirtualenv**
```mkvirtualenv yolo-a-train -r requirements.txt```

## Train enviroment
### YOLOv3 Archive version (DarkNet Architecture)
**pip**
```python3.8 -m pip install -r requirements_YOLOv3_archive.txt```

**mkvirtualenv**
```mkvirtualenv yolo-a-train -r requirements_YOLOv3_archive.txt -p /usr/bin/python3.8```

    
