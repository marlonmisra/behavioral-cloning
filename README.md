#**Behavioral Cloning Project** 

##Introduction##
Udacity created a game that lets you drive around a track and screen-capture the footage and associated steering angles. Using this data, you can train a neural network to predict what steering angles will keep the a car on the road. You can then use test this model on the the car in "autonomous mode". The goal is to get the car to go around one full track without touching the edges of the street. 

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior.
* Build, a convolution neural network in Keras that predicts steering angles from images.
* Train and validate the model with a training and validation set.
* Test that the model successfully drives around track one without leaving the road.
* Summarize the results with a written report.


##Files and testing##

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 containing a video file of a successful lap
* This report (README.md) summarizing the process and results


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```


##Exploratory analysis##

To start with, I looked at the type of data I was dealing with. To summarize, there are:
* 8036 observations (images and associated steering angles)
* Images which each have the dimension 160x320x3 (160 height, 320 width, 3 RGB colors channels)


I then plotted a series of images using matplotlib to get a sense of data quality and data diversity. I found that all the images were informative - although there was a bias towards images with left turns.

![Image plotting](assets/random_images.png =500x50)


##Reading and splitting##
I started by reading in the target files which contain the steering angles and the images. I then did a train/validation split and allocated 95% of the data to training.

##Generator and augmentation##
Then I implemented a Python generator to read in the image files. This was important because my computer doesn't have enough memory to handle reading in all images. 

In addition to the center images, I also read in the left and right images. I applied a 0.05 adjustment to the steering angles for the non-center images to account for the different perspective. I also did image augmentation by reflecting each image and multiplying the angles by -1. This was important because the training data had a bias towards left turns.


##Solution design##
I started with a simple two-layer fully connected network to make sure that I could make predictions. I tested on a  small number of observations, and didn't worry about overfitting. From there, I slowly added layers of complexity to make the model fit better - ensuring that loss decreased as I made changes. 

In terms of preprocessing, I made use of keras functions to do it in the neural network. I added a Lambda layer to normalize inputs and a cropping layer to remove 50 pixels from the top and 25 from the bottom - these seemed to not add any information. 

I added multiple Convolution2D layers with increasing depth and a frames of 5x5 and 3x3.  Then, I added additional dense layers and dropout layers with dropout probability 0.2 and 0.5 to improve the robustness of the model and reduce overfitting. After each convolution layer, dropout layer, or dense layer, I used ELU activations to introduce non-linearity. 

```model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dense(1))
    ```

I used mean squared error as a loss function because this is a common standard that works well for these types of problems. I used the Adam optimizer so that I didn't have to worry about hyperparameter tuning. 

For training, I played around with a number of batch size and epoch combinations to see what worked best. I found that after 10 epochs, improvements generally wore off. I also discovered that batch sizes of 92 worked well (batch size 16 * 6 dimensions of augmentation).


##Testing##
After training the final model, I ran the simulator in autonomous mode and was able to go around the entire track. 


##Closing thoughts##
There are a number of things I could have done to further improve the model. These include:
* I didn't spend a lot of time training or tuning hyperparameters because I was doing the computations on my local machine. For additional accuracy, I would use Amazon EC2 and do training there. 
* I could have collected more data or simulated additional data. I didn't generate any new data by adjusting shadows or brightness or applying translations and rotations. 
* To make turns even more smooth, I could have used a smoothing function that tracks a rolling average of recent steering angles and adjusts them towards their mean.
* The training data I used mostly consisted of good driving. However this introduces a bias and makes the model less good at recovery. For improved accuracy, I could have added additional data specifically focused on recovery (starting from the edges of a lane and returning to the center).


