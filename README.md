# Handwritten-Digit-KNN-CPP

## What is this

A simple handwritten digit classifier in KNN coded with C++ using the MNIST dataset.

There is no optimization on the algorithm.

## Build & Run

The dataset will be downloaded while building, so you just need:

1. Build it with `make`
2. Now if there is no error, you will get a huge binary executable file (~55MB) with the dataset embedding in.
3. Usage: `./knn K P`.
4. Example: `./knn 10 3`, in which the '10' is the 'K' and the '3' is the power in `minkowski_distance`,

## Debug

Run the whole dataset while debugging is unnessary, so you can build it in debug mode:

1. Build it with `make debug`
2. Debug Usage: `./debug K P TrainImageNum TestImageNum`.
3. Example: `./debug 10 3 6000 2000`, in which '6000' is the number of images will be used from training dataset and '2000' is the number of images will be used from testing dataset.

## License

WTFPL
