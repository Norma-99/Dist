import sys
import argparse
from tensorflow.keras.layers import GaussianNoise, Dense, InputLayer
from tensorflow.keras.models import Sequential


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_count', type=int, required=True)
    parser.add_argument('--layer_size', type=int, required=True, nargs='+')
    parser.add_argument('--has_gaussian', type=bool, default=False)
    parser.add_argument('--noise_variance', type=float, required=False)
    args = parser.parse_args()

    model = Sequential([InputLayer(input_shape=(31,))]) #(87,) or (74,)
    if args.has_gaussian:
        model.add(GaussianNoise(args.noise_variance))
    for i in range(args.hidden_count):
        model.add(Dense(args.layer_size[i], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary(print_fn=lambda x: print(x, file=sys.stderr))
    print(model.to_json())
    return 0


if __name__ == '__main__':
    exit(main())
