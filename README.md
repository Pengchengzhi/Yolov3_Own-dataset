# Yolov3_Own-dataset
Train yolov3 on my own dataset, Ubuntu 16.04
## How to train
* Pre requirements
Install openCV, CUDA

Optional but highly recomended, will be a lot faster.
* Download model
0. Download source code
(''')
 git clone https://github.com/pjreddie/darknet
 cd darknet
(''')
There is another edition of [yolov3](https://github.com/AlexeyAB/darknet), chose whatever you want.
1. Change makefile
Want to use GPU
'GPU=1'
Want to use openCV
'OPENCV=1'
Want to use CUDNN
(''')
 CUDNN=1
 NVCC=/usr/local/cuda-10.0/bin/nvcc
(''')
2. Compile
'Make'


