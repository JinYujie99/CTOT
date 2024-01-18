# CTOT

PyTorch Implementation for "Temporal Domain Generalization via Learning Instance-level Evolving Patterns" 

## Usage
This example code include both classificaiton and regression tasks. To run our experiments on 2-Moons dataset, go to the `classification` folder. To run our experiments on House dataset, go to the `regression` folder. 


1.Pretraining

```
bash scripts/pretrain.sh
```
We have also provided our pre-trained models in the `models` folder.

2.Gnenerate Trajectory Data using OT

```
bash scripts/gen_tra_data.sh
```
We have also provided our generated trajectory data in the `data` folder.

3.Train the continuous-time model and make predictions for the target domain

```
bash scripts/train.sh
```

## Requirements
Environments used in our experiments:
* Python 3.8.18
* PyTorch 1.9.1
* Numpy 1.24.3
* torchvision 0.9.1
* POT 0.9.1
* torchsde 0.2.6


