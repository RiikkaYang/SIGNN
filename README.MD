# Sociologically-Informed Graph Neural Network for Event Prediction
Source code for our paper: Sociologically-Informed Graph Neural Network for Event Prediction

## Requirements
- Python 3.10
- numpy == 1.19.5
- pandas == 1.2.0
- scikit_learn == 1.0.2
- scipy == 1.6.0
- torch == 1.9.0
- torchvision == 0.9.1
- transformers == 4.8.2

## How to Use
To train and evaluate the model with specific parameters, run the following command:
```
python3 main_signn.py --method SIGNN --dataset GMF --save_dir output/ --num_hidden_layers 10 --hidden_features 8 --alpha 0.1 --beta 0.1 --num_epochs 1000 --lr 0.001 --type_odm SBCM
```
