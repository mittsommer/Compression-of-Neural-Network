import os

l = [1, 0.9999, 0.999, 0.99, 0.9, 0.8]

for i in l:
    os.system("python main.py --l {}".format(i))