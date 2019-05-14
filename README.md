# Train yolov3 on my own dataset
Ubuntu 16.04
## How to train
### Pre requirements

* Install openCV, CUDA

* Optional but highly recommended, will be a lot faster.
### Download model and try a test
**0. Download source code**

```
 git clone https://github.com/pjreddie/darknet
 
 cd darknet
```

Instead of this original edition yolov3, there is another AB edition of [yolov3](https://github.com/AlexeyAB/darknet), chose whatever you want.

**1. Change makefile**

* Want to use GPU  --->   `GPU=1`

* Want to use openCV   --->   `OPENCV=1`

* Want to use CUDNN

```
 CUDNN=1
 
 NVCC=/usr/local/cuda-10.0/bin/nvcc
```

**2. Compile**  --->   `make`

**3. Test**

Download a pre-trained model to test

`wget https://pjreddie.com/media/files/yolov3.weights`

Then test the dog picture

`./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg`

If you are lucky, you can see the following picture.

![dog_picture](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/dog_predictions.jpg)

### Customize my dataset

**0. Background**

My project is to detect five different kinds of objects: `lizard`,`bird`,`car`,`dog`,`turtle` and I use [labeling](https://github.com/tzutalin/labelImg) to label my pictures. After that, prepare a folder to save all the pictures and another folder to save all the `.xml` documents.

**1. Generate .txt file**

 `.txt` file that I prepared: `train.txt`,`val.txt`and a lot of `picture_name.txt` in folder `labels`.

`train.txt`: Store the paths of pictures used for train(without .jpg). One path per line.

`val.txt`: Store the paths of pictures used for test(without .jpg). One path per line.

`picture_name.txt`: Store object information in the picture, one object per line. It should be customized to 

> (object-class) (x_center) (y_center) (width) (height)

for example:

> 0 0.66015625 0.28515625 0.6171875 0.5390625

* Put folder `labels` and folder `pictures` under the same directory.

**2. Change some relevant files and codes**
* Copy `voc.names` ,rename to `animal.names`, and change it to my class names, one class name per line. 
```
bird
car
lizard
dog
turtle
```
The order of the names should match class index in `picture_name.txt`.

* Copy `voc.data`, rename to `animal.data`.
```
 classes =  5 # My class number
 train   =  data/train.txt # Path to train.txt
 valid   =  data/val.txt # Path to val.txt
 names   =  data/animal.names # Path to animal.names
 backup  =  backup/
```

* Copy `yolov3-voc.cfg` in folder `cfg`, rename to `animal.cfg`, make the following changes:
```
[net]
# Testing
# batch=1  # Use it when you test the model
# subdivisions=1    # Use it when you test the model
# Training
batch=64
subdivisions=32     # Set it smaller if memory full
...              
...
...
learning_rate=0.001  
burn_in=1000
max_batches = 30000  # Change according to your needs
policy=steps
steps=10000,20000    # Change with max_batches
...
...
...
[convolutional]
size=1
stride=1
pad=1
filters=30         # filters = 3*(classes + 5)
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=5          # change to your class number
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1           # set to 0 if memory full
...
...
...

# There are three [yolo] layers in total, change all of them.
# Behind the three [yolo] layers there is a [convolutional] layer, change their filters.
```

**3. After the change**

I have 150 pictures for training and 30 pictures for testing, each group, so my folder `data` is like 

>train_images
>>`000001.JPEG`...`000750.JPEG`
>
>train_labels
>>`000001.txt`...`000750.txt`
>
>val_images
>>`000001.JPEG`...`000150.JPEG`
>
>val_labels
>>`000001.txt`...`000150.txt`
>
>animal.data
>
>animal.names
>
>train.txt
>
>val.txt

**4. Download pre-trained weights**
`wget https://pjreddie.com/media/files/darknet53.conv.74`

**5. Start training**

`sudo ./darknet detector train data/animal.data cfg/animal.cfg darknet53.conv.74 `

When I finished training and want to a picture of loss, I find I did't save it. So maybe try the following code for training instead of the previous one:

`sudo ./darknet detector train pds/fish/cfg/fish.data pds/fish/cfg/yolov3-fish.cfg darknet53.conv.74 2>1 | tee visualization/train_yolov3.log `

### Test the result

**Test a single picture**

`sudo ./darknet detector test data/animal.data ./cfg/animal.cfg ./backup/animal_20000.weights ./data/val_images/000133.JPEG`

My results:

![dog](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc3.jpg) ![car](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc1.jpg) ![lizard](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc2.jpg) ![turtle](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc5.jpg) ![bird](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc4.jpg)







