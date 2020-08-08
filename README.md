# **Behavioral Cloning** 

## Writeup Report

### Nikko Sadural

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_images/center_ccw.jpg "CCW Lap Image"
[image2]: ./report_images/center_cw.jpg "CW Lap Image"
[image3]: ./report_images/left_recover.jpg "Recovery Image"
[image4]: ./report_images/turn_recover.jpg "Turn Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 showing a successful lap around track one in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolutional neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of the following layers:

* Input lambda layer to normalize the image with zero-mean and maintain image shape (160, 320, 3) (model.py line 68).
* Inception network with 311 layers, including convolution, batch normalization, max pooling, and average pooling layers with the normalized input shape (160, 320, 3) and output shape (3, 8, 2048) (model.py line 69).
* Global average pooling layer to reduce shape to 2048 and reduce overfitting, followed by a ReLU activation layer (model.py line 70).
* Dense layer to reduce the output shape to 512 (model.py line 71).
* Dense output layer to reduce the final output shape to 1 since we are predicting a single steering angle (model.py line 72).

#### 2. Attempts to reduce overfitting in the model

The model contains global average pooling layer to help reduce overfitting (model.py line 70).

The model was trained and validated using a training/validation split of 70%/30% (model.py line 83). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 75).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and recovering specifically from difficult turns with red/white roadside markings. I also augmented the data to increase the size of the training/validation set by flipping the recorded images and respective steering angle measurements (model.py lines 35-39).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to utilize transfer learning of the ImageNet-pretrained InceptionV3 neural network with unfrozen weights to train all layers. The input is normalized with zero-mean, and the output shape is reduced down to a shape of 1 since the simulation requires 1 steering angle prediction to drive autonomously for a given image.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (70%/30% respectively). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it included a global average pooling layer after the Inception network layer. Following the pooling layer are the dense layer to reduce the shape to 512 and finally an output layer of size 1 containing the predicted steering angle for the given image.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, mostly on the difficult turns with red/white roadside markings. To improve the driving behavior in these cases, I collected more favorable full-lap driving data and failed-turn recovery maneuvers. After several rounds of additional data collection, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 66-72) consisted of a convolutional neural network containing a normalizing input lambda layer, the InceptionV3 network layer, a global average pooling layer, and two densely connected layers. The network has an input shape of (160, 320, 3) and an output shape of (1).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one with acceptable center-lane driving in the counter-clockwise direction. Then I recorded another acceptable lap in the clockwise direction such that the model is not biased for predicting left turns:

![alt text][image1]
![alt text][image2]

I then recorded the vehicle recovering from the left and right sides of the road back to center so that the vehicle would learn to return to center if it were to veer off to the left or right sides of the road. These images show what a recovery looks like starting from the sides of the road for normal driving and during the sharp turns :

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would help generalize the dataset.

After the collection process, I had 6097 data points from the vehicle's center camera. I then preprocessed this data by dividing by 255.0 and subtracting by 0.5 to normalize each image with zero-mean.

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by a huge decrease in the training and validation losses by the 3rd epoch and marginal decrease afterwards. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
