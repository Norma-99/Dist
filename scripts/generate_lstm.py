import sys
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM,Reshape, InputLayer, Flatten


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_count', type=int, required=True)
    parser.add_argument('--layer_size', type=int, required=True, nargs='+')
    parser.add_argument('--has_gaussian', type=bool, default=False)
    parser.add_argument('--noise_variance', type=float, required=False)
    args = parser.parse_args()

    model = Sequential([InputLayer(input_shape=(31,1,))]) 
    if args.has_gaussian:
        model.add(GaussianNoise(args.noise_variance))
    for i in range(args.hidden_count):
        model.add(LSTM(args.layer_size[i], return_sequences=True))
    model.add(Dense(1, activation='sigmoid'))
    model.summary(print_fn=lambda x: print(x, file=sys.stderr))
    print(model.to_json())
    return 0


if __name__ == '__main__':
    exit(main())
