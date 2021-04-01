sudo apt install python3.7
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
sudo update-alternatives --config python3sudo update-alternatives --config python3
sudo apt install python3.7-venv
sudo apt install python3-venv
python3 -m venv tutorial-env
source tutorial-env/bin/activate
pip install --upgrade pip
python3 -m pip install --upgrade setuptools

sudo apt-get remove python3-apt
sudo apt-get install python3-apt

echo "deb http://repo.yandex.ru/yandex-disk/deb/ stable main" | sudo tee -a /etc/apt/sources.list.d/yandex-disk.list > /dev/null && wget http://repo.yandex.ru/yandex-disk/YANDEX-DISK-KEY.GPG -O- | sudo apt-key add - && sudo apt-get update && sudo apt-get install -y yandex-disk

sudo apt update
sudo apt-get install software-properties-common
