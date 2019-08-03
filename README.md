# Train yolov3 on my own dataset
Environment: Ubuntu 16.04 

Paper: [YOLOv3.pdf](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

## How to train
### Pre-requirements

* OpenCV

* CUDA 

Otional but highly recommended, will be a lot faster.

### Download model and try a test
**0. Download source code**

```
 git clone https://github.com/pjreddie/darknet
 cd darknet
```

Instead of this original edition of yolov3, there is another [AlexeyAB edition](https://github.com/AlexeyAB/darknet) of yolov3, chose whatever you want.

**1. Change makefile** ---> `vi Makefile`

* Want to use GPU  --->   `GPU=1`

* Want to use openCV   --->   `OPENCV=1`

* Want to use CUDNN

```
 CUDNN=1
 NVCC=/usr/local/cuda-10.0/bin/nvcc
```

**2. Compile**  --->   `make`

**3. Test**

Download a pre-trained model to test --> `wget https://pjreddie.com/media/files/yolov3.weights`

Then test the dog picture ---> ` ./darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg`

If you are lucky, you will get a picture like this

![dog_picture](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/dog_predictions.jpg)

### Customize the dataset

**0. Background**

My project is to detect five different kinds of objects: `lizard`,`bird`,`car`,`dog`,`turtle` and I use [labelImg](https://github.com/tzutalin/labelImg) to label my pictures. After that, prepare a folder to save all the pictures and another folder to save all the `.xml` documents. 

Do not delete these `xml` files, they are needed for computing mAP.

**1. Generate .txt file**

`.txt` file that I prepared: `train.txt`,`val.txt`and a lot of `picture_name.txt` in folder `labels`.

`train.txt`: Store the paths of pictures used for training (without .jpg). One path per line.

`val.txt`: Store the paths of pictures used for testing (without .jpg). One path per line.

`picture_name.txt`: Store bounding box information of the picture, one object per line. It should be customized to 

> (object-class) (x_center) (y_center) (width) (height)

for example:

> 0 0.66015625 0.28515625 0.6171875 0.5390625

* Put folder `labels` and folder `pictures` under the same directory.

**2. Change some relevant files and codes**
* Copy `voc.names` , rename to `animal.names`, and change the class names, one class name per line. 
```
bird
car
lizard
dog
turtle
```
The order of the names should match class name index in `picture_name.txt`.

* Copy `voc.data`, rename to `animal.data`.
```
 classes =  5                  # Class number
 train   =  data/train.txt     # Path to train.txt
 valid   =  data/val.txt       # Path to val.txt
 names   =  data/animal.names  # Path to animal.names
 backup  =  backup/
```

* Copy `yolov3-voc.cfg` in folder `cfg`, rename to `animal.cfg`, make the following changes:
```
[net]
# Testing
# batch=1            # Use it when you test the model
# subdivisions=1     # Use it when you test the model
# Training
batch=64
subdivisions=32      # Set it bigger if memory full
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
filters=30           # filters = 3*(classes + 5)
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=5            # change to your class number
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1             # set to 0 if memory full
...
...
...

# There are three [yolo] layers in total, change all of them.
# In front of every [yolo] layer there is a [convolutional] layer, change filter number.
```

**3. How it looks like after the change**

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

**4. Download a pre-trained weight** ---> ` wget https://pjreddie.com/media/files/darknet53.conv.74`

**5. Start training** ---> ` sudo ./darknet detector train data/animal.data cfg/animal.cfg darknet53.conv.74` 

When I finished training and want to visulize the loss, I found I did't save it. So maybe try the following code for training instead of the previous one:

` sudo ./darknet detector train data/animal.data cfg/animal.cfg darknet53.conv.74 2>1 | tee visualization/animal.log `

### Test the result

**0. Detect a single picture**

* Switch the `.cfg` file to test mode:

```
[net]
# Testing
batch=1  # Use it when you test the model
subdivisions=1    # Use it when you test the model
# Training
# batch=64
# subdivisions=32     # Set it smaller if memory full
...    
...
...
```
* Recompile:

`make clean` and `make`

* Now choose a picture from test set and test:

`sudo ./darknet detector test data/animal.data ./cfg/animal.cfg ./backup/animal_20000.weights ./data/val_images/000133.JPEG`

I trained a total number of 128w pictures, my results:

![car](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc1.jpg) ![lizard](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc2.jpg) ![dog](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc3.jpg) ![turtle](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc5.jpg) ![bird](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/suc4.jpg)

**1. Compute Recall and IoU**
* Definition:

True posotive ---> tp

True negative ---> tn

False positive ---> fp

False negative ---> fn

Precesion = tp/(tp+fp) 

Recall = tp/(tp+fn)

* Compute

Find function `validiate_detector_recall` in `examples/detector.c`, make two changes:

1' Change the path: `list *plist = get _path("data/val.txt")  // to my path to val.txt`

2' Change the `for` loop as directed by [fix error in validate_detector_recall #952](https://github.com/pjreddie/darknet/pull/952/commits/6c8ed1bde84b27e9fbc0259e4ffd415a7d2c951b)

```
++total;
box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
float best_iou = 0;
// for(k = 0; k < l.w*l.h*l.n; ++k){
for(k = 0; k < nboxes; ++k){  // Use this line
    float iou = box_iou(dets[k].bbox, t);
    if(dets[k].objectness > thresh && iou > best_iou){
        best_iou = iou;
```

`make clean` and `make`

`./darknet detector recall data/animal.data cfg/animal.cfg backup/animal_20000.weights`

* My Results: 

On test set ---> RPs/Img: 1.37	IOU: 71.64%	Recall:90.00%

On train set ---> RPs/Img: 1.03	IOU: 91.60%	Recall:100.00%

**2. Compute AP and mAP**
* Definition

P (Precision) ---> for one image in one class

AP (AveragePrecision) ---> for all images in one class

mAP (MeanAveragePrecision) ---> for all images in all classes

* Compute

1' Detect test set and save the results `classname.txt` to `darknet/results` folder.

`./darknet detector valid data/animal.data cfg/animal.cfg backup/animal.backup -out ""`

2' Download [`voc_eval.py`](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py)

3' Build `compute_map.py`:

```
from voc_eval import voc_eval

rec,prec,ap = voc_eval('/home/karry/Documents/Yolo/darknet/results/{}.txt', '/home/karry/Documents/Yolo/darknet/val_xml/{}.xml', '/home/karry/Documents/Yolo/darknet/val.txt', 'classname','.')

print('ap',ap)
```

The first path is for the detection results in the first step.

The second path is for `.xml` documents of test set.

The third path is for `val.txt` which contains picture name of test set, no picture path, no `.jpg`, only picture name. One name per line.

The forth one is class name.

The fifth one is the path to save `annots.pkl` file.

4' Run `compute_map.py`.

Get AP for this class.

* My results

| Class | AP |
| :---: | :---: |
| Dog | 0.8522680776014109 |
| Car | 0.8850865226727296 |
| Bird | 0.7822037102682264 |
| Lizard | 0.9223655913978495 |
| Turtle | 0.7221268965829435 |
| mAP | 0.8328 |

### A debug experience

**0. Problem description**

When I run the test code for a single image, I got pictures with correct lable but no bbox:

![car](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/fail1.jpg) ![lizard](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/fail2.jpg) ![dog](https://github.com/Pengchengzhi/Yolov3_Own-dataset/blob/master/images/fail3.jpg)

**1. How I solved it**

Open program `src/image.c`, find function `draw_box_width()`, change the loop function:

from

>  for(i = 0; i < w; i++)

to

>  for(i = 0; i <= w; i++)

Then recompile --->  `make clean` and `make`

Run the detection command:

` sudo ./darknet detector test data/animal.data ./cfg/animal.cfg ./backup/animal_20000.weights ./data/val_images/000133.JPEG`

Problem solved.

**2. Why it happens**

Generally it's because my picture is too small, only 128 x 128 pixels.

When I use the test code, I'm actually calling function `test_detector()` in `example/detector.c`, and he will call `draw_detections()` defined in program `src/image.c`

```
void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
    int i,j;

    for(i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int class = -1;
        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j] > thresh){
                if (class < 0) {
                    strcat(labelstr, names[j]);
                    class = j;
                } else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[j]);
                }
                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
            }
        }
        if(class >= 0){
            int width = im.h * .006;
            ...
            ...
            ...
```

In the function above, `im.h` = 128 so `width` will be 0. And then he will pass these parameters to `draw_box_width()`

```
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i=0;
    for(i = 0; i < w; i++){
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

```

At that time, `i` = 0, `w` = 0, the `for` loop won't run,  he won't call `draw_box()` so I can't get my boundingbox. By adding a `=` can make the program run and that's how I solved the problem.

* Pictures with their `im.h` smaller than 167 can have the same problem. 










