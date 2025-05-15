# Perona Malik
Applies Perona Malik diffusion step-by-step to any digital image. Also includes code to apply Gaussian smoothing to any digital image.

## Install 
- Clone the repository (run "git clone (repository url)") and change directories into perona_malik <br />
- Install dependencies by running:
```
    pip install -r requirements.txt
```


## Run Perona Malik
```
python gui.py
```

The GUI should look like the image below: <br />
<img width="764" alt="Screenshot 2023-08-11 at 12 57 46 AM" src="https://github.com/remi-kuba/perona_malik/readme_images/gui.png"> <br />

## Run Gaussian smoothing
- Place the desired image file into the images folder
- Edit the name of the image file in gaussian.py and the amount of blur on lines 22 and 25
- Run ```python gaussian.py``` and the resulting image will be saved in the results folder

## Example Result
Noisy image input:

<img width="764" alt="Screenshot 2023-08-11 at 12 57 46 AM" src="https://github.com/remi-kuba/perona_malik/readme_images/noisy_img1.png"> <br />

Gaussian blur output:

<img width="764" alt="Screenshot 2023-08-11 at 12 57 46 AM" src="https://github.com/remi-kuba/perona_malik/readme_images/gaussian_noisy_img1.png"> <br />

Perona-Malik diffusion output:

<img width="764" alt="Screenshot 2023-08-11 at 12 57 46 AM" src="https://github.com/remi-kuba/perona_malik/readme_images/pm_noisy_img1.png"> <br />