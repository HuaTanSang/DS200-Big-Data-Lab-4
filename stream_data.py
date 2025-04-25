import argparse
from dataset_streamer.streamer import DatasetStreamer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream FashionMNIST in epochs")
    parser.add_argument('--images', '-i', required=True, help="Path to train-images-idx3-ubyte")
    parser.add_argument('--labels', '-l', required=True, help="Path to train-labels-idx1-ubyte")
    parser.add_argument('--batch-size', '-b', type=int, default=100)
    parser.add_argument('--sleep', '-t', type=float, default=2.0)
    parser.add_argument('--epochs', '-e', type=int, default=1)
    args = parser.parse_args()

    streamer = DatasetStreamer(args.images, args.labels, host='localhost', port=6100)
    streamer.stream_epochs(
        batch_size=args.batch_size,
        sleep_time=args.sleep,
        epochs=args.epochs
    )





