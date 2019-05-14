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

`picture_name.txt`: Store object information in the picture, one object per line.

* Put folder labels and folder pictures under the same directory.

I have 150 pictures for training and 30 pictures for testing, each group, so my file is like 

>train_images
>>`000001.jpg`...`000750.jpg`
>
>train_labels
>>`000001.txt`...`000750.txt`
>
>val_images
>>`000001.jpg`...`000150.jpg`
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

**2. Change relevant codes**




