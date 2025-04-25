# DS200 Big Data Lab 4
## Introduction 
This repository is to simulate the server sending data to another server to train and evaluate model, especially on FashionMNIST with LeNet. 

## Prerequisite
`Ubuntu 22.04.5 LTS`   
`Python 3.10.12` 

## Installation 
```
git clone https://github.com/HuaTanSang/DS200-Big-Data-Lab-4.git
cd DS-200-Lab-4
``` 
Open terminal and install requirements 
```
pip3 install requirements.txt  
```

Then run the stream_data.py to start server and wait for data 
``` 
PYTHONPATH=. python3 stream_data.py \
  --images /path/to/train-images-ubyte \
  --labels /path/to/train-labels-ubyte \
  --batch-size 100 \
  --sleep 0.5 \
  --epochs 5
```
Open another terminal to start client and send data 
```
spark-submit --master local[2] main.py
```

