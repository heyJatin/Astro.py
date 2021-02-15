# Astro.py
Asteroid Detection in .fits files using Tensorflow Keras.

## How to detect
1. Organize your dataset into three folders
  ```
 >data
    >detect
      >set_of_fits
    >training
      >Asteroids
      >NotAsteroids
    >validation
      >Asteroids
      >NotAsteroids
  ```
  
  2. Set parameters according to your dataset in ```astro.py```
  
  3. Run ```python astro.py```
  
  ## How it works
  
  I've explained it's working in this YouTube video: https://youtu.be/QZC7bWgOmfo
  
  1. Convert .fits to binary PNG using value of x (experimented). 
    Checkout these images for different values of x
     * [x=10 (too noisy)](https://github.com/heyJatin/Astro.py/blob/main/example_img/x%3D10.png)
     * [x=12 (perfect)](https://github.com/heyJatin/Astro.py/blob/main/example_img/x%3D12.png)
     * [x=64 (data lost)](https://github.com/heyJatin/Astro.py/blob/main/example_img/x%3D64.png)
     
  2. Images of same sets are subtracted from each other, and then merged. This will eliminate static objects, only moving ones will be left visible.  
  [Example](https://github.com/heyJatin/Astro.py/blob/main/example_img/sub.png)
  
  3. Noises left after 2nd step is further reduced using cv2.  
  [Example](https://github.com/heyJatin/Astro.py/blob/main/example_img/denoised.png)
  
  4. After processing all images, training of our model will start. We're using **Tensorflow Keras, VGG16 'imagenet'**
  
  5. After training completes, model will start detecting Asteroids from Testing set ```data/detect```
  
  ---
  
  ### Contribution:
  This is a basic level project, with enough contributions, we can further scale it to greater heights.
