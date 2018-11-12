# Chrome Dino

![alt text](images/chrome_dino_illustration.png "Chrome Dino Illustration")

## Install requirements

To install requirements, you can run `pip install -r requirements.txt`

## Selenium

In this repository, Selenium requires a Chrome Driver. 
1. You can download it [here](https://sites.google.com/a/chromium.org/chromedriver/downloads)
2. Click on "Latest Release: ChromeDriver x.xx"
3. Download the driver corresponding to your OS
4. Unzip the archive
5. Move the driver in the folder of the repository. Otherwise, you can update the driver's path in `dino.py`

## OpenCV

You might encounter issues with OpenCV, like I did on Mac OS El Capitan.
OpenCV 2.4.0.11 and 3.4.0.12 dit not work (well). I chose to install the version 3.3.0.10
You might have to choose a custom version, depending on your OS.

To do so, you can run `pip install opencv-python==x.x.x.xx`, replacing "x.x.x.xx" by the name of the desired version.
