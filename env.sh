sudo apt install python3.7
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
sudo update-alternatives --config python3sudo update-alternatives --config python3
sudo apt install python3.7-venv
python3 -m venv tutorial-env
source tutorial-env/bin/activate
pip install --upgrade pip
python3 -m pip install --upgrade setuptools
