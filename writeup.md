# **Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[image0]: ./examples/laneLines_thirdPass.jpg "LaneLines"

[image1]: ./examples/grayscale.jpg "Grayscale"

[image2]: ./examples/line-segments-example.jpg "Line Segments"

---

### **Reflection**

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps.

 1. *Convert to Grayscale*:

    First thing we want perform is to reduce the amount of data to be processed. Image consist of 3-channels (Red, Green and Blue). As to build simple lane detection pipeline we do not really need to have the color information, hence we will convert this to Grayscale, where each pixel consist of value from 0 to 255, where 0 implies complete black and 255 implies to complete white. Hence grayscale image consist of 1-channel having gray pixel values. This has reduced our complexity from the 3-channel data to 1-channel data, and at the same time we are preserving image properties such as object shapes and loosing out only the color information.

    ![alt text][image1]

 2. *Apply Noise Filter*:

    Here, we will apply Guassian Noise Filter in order remove unwanted noise from the image and make the image little blury and smoothness. This will help in finding the correct edge points.

    In current pipeline, I have used 5x5 kernel for performing Guassian Smoothing.

 3. *Find Edges*:

    In order to find the lane from the grayscale image, I have performed Canny Edge detection which internally computes gradients of each pixels with respect to x, y dimensions and performs maximul-suppression to avoid false-positive edges. Canny edge detector takes in two type of threshold, low threshold and high threshold. All the detected gradient pixels whose intesities are below low threshold are simply considerred to be non edge pixel and discarded where as pixels having intensities greater than high threshold are edge pixels. For those pixels whose intensity lies between low threshold and high threshold, they are being eveluated to be edge pixel if and only if their connected neighbouring pixel is edge pixel.

    In current pipeline, I have used low_thresh:230 and high_thresh:250 to obtain good results.

 4. *Select Region Of Interest*:

    Lane markings are always going to be placed on bottom half of the image (assuming camera is not inverted) and hence this could be our area of interest. As images are 2D (i.e. it does not have information on depth), but considering the camera placed on car, our lanes will converge at some point on center of the image. This point is called Vanishing point.

    In current pipeline, I have assumed that vanishing point is slightly below the center of the image (i.e. y = height * 0.6) and considering this assumption, drawn region of interest to be trapeziod.

    Trapeziod: (w * 0.05, h * 0.95), (w * 0.40, h * 0.60), (w * 0.60, h * 0.60), (w * 0.95, h * 0.95)

    With selected area of interest, I have masked the edges and considered only those edges which are inside this area.

 5. *Find Lines with Hough Transform*:

    Once we have edges filtered by selecting edges fall under ROI (Region of interest), I can perform Hough Transform for finding all the line points.

    With these line points, if slope of the line is -ve, line is on the left side else on right side. This is because of the nature of convergence os lines to vanishing point. All the lines which has slope = 0 or infinity should be ignored as they are not going to be lane lines.

    ![alt text][image2]

 6. *Average/Extrapolate found Lane Lines*:

    Once we have left, right Lines and their respective slopes, we can find the average slope of the left and right lines. This will give us unified average line segment. Here we also need to take consideration of the 'c' from (y = mx + c) which adds intercept information to the lines.

    With ``y = mx + c`` equation, where we have ``m`` and ``c`` for each of the line, we can extrapolate to average line with same slope ``m`` with ``c``. This will be our Average Lane.

#### Modification to `draw_lines()` function

Existing ``draw_lines()`` draws raw lines on the image where as we want lane line to be consistant and straight. For this I have used line equation ``y = mx + c`` for finding slope `m` and intercept `c` for each of the lines, where if slope is -ve, I considered this to be left line, and for +ve, right line. For all those lines whose slope `m` is `0` or `inf`, I ignore them as they can't be lane lines.

Once I have left/right lines, I had performed intercept adjustment by performing dot product of (slope, intercept) to (length) of each line and divided them by the total of the line length. This had given me extrapolated line slope, intercept.

With the assumption that Lane lines converge at center of the image, I considered lowest Y of the lane line would be 60% of the height. Hence with this information and `y = mx + c` I found x1, y1, x2, y2 for the left and right lanes.
```
y1 = height
y2 = height * 0.6
x1 = (y1 - intercept) / slope
x2 = (y2 - intercept) / slope
```

Finally I used `cv2.line` for drawing lines from (x1, y1) to (x2, y2) for left and right Lanes. Which looks like --

![alt text][image0]

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be when there is no lane markings on the road, current pipeline fails. The reason behind this is obvious that we donot have edges captured from the road and hence no lines from hough transform.

Another shortcoming could be, when there is a curve on the lane, this fails to track curviness of the lane as we are extrapolating/averaging line into line polygon and hence we have final result will be lane line based on the average slope we received for all the lane lines.

There is one more shortcoming, as it takes region of interest in account, this unable to find other lanes on the road appart from on which car is moving.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use of `fitPoly()` to find best fit curve for all the found Lane Lines, to provide smooth curved lanes when they are curvy.

Another potential improvement could be to use of dnn based model for detecting lanes which will eliminate use of region of interest and also could find all the lanes visible, hence could be used to decide for lane switching too. Although there is one shortcoming to this, there are no such labeled dataset for this available. But this can be autogenerated by current pipeline, and later fed to any dnn model of object detection.