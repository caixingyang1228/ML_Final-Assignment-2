# Machine Learning Assignment 2: Classification, images  
 ---

## Problem stament  

>Flying Dollar Airport is a small, public-use airport in the Poconos with a 2,400 foot turf runway. A Nest Cam is pointed at a 200 foot section in the middle of the runway. When the Nest Cam detects activity, a static image is captured. “Activity” is any significant movement in the frame. This includes aircraft arrivals and departures, but also includes many other things such as cloud movement, animals, rain, snow, trees moving in the wind, people, vehicles, the movement of light, and other motion. Of the 6,758 images, only 101 contain aircraft.  

We are going to train a new model using Histogram of Oriented Gradients (HOG) to compare with the exsiting model 'Canny algorithm'.  

## Histogram of Oriented Gradients (HOG)  

In our original notebook, we have seen Canny edge detection, which is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed[(source)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) for the use of downscaling images. 
The Histogram of Oriented Gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection[(source)](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients). A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information[(source)](https://www.learnopencv.com/histogram-of-oriented-gradients/). Here we are going to compare the accuracy of Canny algorithm and HOG.  

##### feature.hog  

We have already had the build-in library `feature.hog` for extracting Histogram of Oriented Gradients (HOG) for a given image.
First we import the library:  

```
from skimage.feature import hog  
```
We suppose to crop and resize the original image, since we already have the cropped images, we only need to resize them to 64 * 128 by using `transform.resize(img_raw, (64, 128))`. Here is our code in the notebook:  

```sh
def image_manipulation(imname, imgs_path, imview=False):
    warnings.filterwarnings('ignore')
    imname = imgs_path + imname + '.png'
    img_raw = io.imread(imname, as_gray=True)
    resized = transform.resize(img_raw, (64, 128)) # resize image

    final_image = feature.hog(resized, 
                              orientations=9, 
                              pixels_per_cell=(16, 16), 
                              cells_per_block=(4, 4))
    
    if imview==True:
        io.imshow(final_image)
    warnings.filterwarnings('always')
    return final_image
```

From the HOG document of Jupyter notebook, we could check the **default setting** of HOG:  

```sh
feature.hog(
    image,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    block_norm='L2-Hys',
    visualize=False,
    visualise=None,
    transform_sqrt=False,
    feature_vector=True,
    multichannel=None)
```

##### Parameters  

**image : (M, N[, C]) ndarray**  
    Input image.  
**orientations : int, optional**  
    Number of orientation bins.  
**pixels_per_cell : 2-tuple (int, int), optional**  
    Size (in pixels) of a cell. We choose 16 * 16.  
**cells_per_block : 2-tuple (int, int), optional**  
    Number of cells in each block. We choose 4 * 4.  
**block_norm : str {'L1', 'L1-sqrt', 'L2', 'L2-Hys'}, optional**  

| Block normalization | Method |
| --- | --- |
| L1 | Normalization using L1-norm. |
| L1-sqrt| Normalization using L1-norm, followed by square root. |
| L2| Normalization using L2-norm. |
| L2-Hys | Normalization using L2-norm, followed by limiting the maximum values to 0.2 ('Hys' stands for 'hysteresis') and renormalization using L2-norm. (default) | 

**visualize : bool, optional**  
     Also return an image of the HOG.  For each cell and orientation bin, the image contains a line segment that is centered at the cell center, is perpendicular to the midpoint of the range of angles spanned by the orientation bin, and has intensity proportional to the corresponding histogram value.  
**transform_sqrt : bool, optional**  
    Apply power law compression to normalize the image before processing. DO NOT use this if the image contains negative values.
**feature_vector : bool, optional**  
    Return the data as a feature vector by calling .ravel() on the result just before returning.  
**multichannel : boolean, optional**  
    If True, the last 'image' dimension is considered as a color channel, otherwise as spatial.  

##### Returns  

out : (n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient) ndarray  
    HOG descriptor for the image. If `feature_vector` is True, a 1D (flattened) array is returned.  
hog_image : (M, N) ndarray, optional  
    A visualisation of the HOG image. Only provided if `visualize` is True.  
    
##### Comparison:  

Canny algorithm:  
![](/images/before.png)  
Histogram of Oriented Gradients (HOG):  
![](/images/after.png)  

## Conclusion  

We significantly increased the prediction accuracy.  

## References  
---
 - [Histogram of Oriented Gradients](https://www.learnopencv.com/histogram-of-oriented-gradients/)  
 - [HOG Document at scikit-image.org](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.hog)  
 - [Wiki page of "Histogram of oriented gradients"](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)  
 - [Wiki page of "Canny edge detector"](https://en.wikipedia.org/wiki/Canny_edge_detector)  
