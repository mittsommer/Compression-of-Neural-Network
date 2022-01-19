# Compression-of-Neural-Network
## requirements
- Python 3.7+
- Pytotch 1.7
## usage
- put data-set in folder ```./data``` 
- training on GPU
```
python main.py --gpu --batch_size 1024 --learning_rate 0.001 --epoch 30 --l 0.9999 --dataset MNIST --model LeNet3_3 --time 1
```
- quantization
```
python quantiz.py --gpu --batch_size 1024 --learning_rate 0.001 --epoch 30 --l 0.95 --dataset FashionMNIST --model LeNet3_3 --time 1
```
