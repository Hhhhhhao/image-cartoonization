# Image-Cartoonization


18-786 Course Project

## Installation

First install required packages:
```
conda create --name cartoon --file requirements.txt
conda activate cartoon
```

Then switch to branch that you will be working on:
```
git checkout [-b] cyclegan/cartoongan/whitebox/
```

**To create trainer routine, create a trainer named 'name_trainer.py' in trainers and implement the necessary functions.**
After you making changes and debugging them to ensure successful training, create to pull request to master so that we can merge.
 

## Training
To train a model:
```
CUDA_VISIBLE_DEVICES=7,8 python main.py --exp-name cartoongan/cyclegan/whitebox --data-dir /home/zhaobin/cartoon/ --n-gpu 2 --tensorboard --num-workers 8 --src-style real --tar-style gongqijun/tangqian/xinhaicheng/disney --epochs 200 --batch-size 32 --g-lr 1e-4 --d-lr 2e-4 --adv-criterion LSGAN --lambda-adv 1.0 --lambda-rec 5.0 --image-size 128 --down-size 16 --num-res 4 --data-aug-policy  color,translation,cutout/translation,cutout[for whitebox]
```

## Evaluation
To evaluation a model on image size 256:
```
CUDA_VISIBLE_DEVICES=5 python eval.py --checkpoint-path expoeriments/exp/checkpoints/xxx --image-size 256
```
And see the results in 'expoeriments/exp/results'


## Results

 
