# Cracking Arbitrary Substitution Ciphers with Statistics-Informed Seq2Seq Models

## Installation

```
pip install torch datasets
```

## Usage

To tarin a decipherment model, use our `train_decipher_model.py`. Check the python file to for a detailed explanation of what each argument to the program stands for.

```
python train_decipher_model.py 
```

To test a trained model, you can use `evaluate_model.py` for quantitative evaluation with symbolic accuracy or `demo.ipynb` for a more qualitative inspection on the outputs of the models. 