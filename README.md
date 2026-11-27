# CIFAR-10 Image Prediction

A convolutional neural network that classifies [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) images
(60,000 32×32 colour images in 10 classes), built as part of the SmartKnower program.

## What it does

1. Downloads `cifar-10-python.tar.gz` (if it isn't there yet) and extracts it.
2. Shows stats and a sample image from a batch.
3. Normalises the pixels, one-hot encodes the labels, and pickles the preprocessed batches plus a validation set.
4. Trains a 4-block CNN (conv → max-pool → batch norm, then four dense layers with dropout) with Adam for 10 epochs,
   batch size 128, keep probability 0.7, learning rate 0.001.
5. Saves the model to `./image_classification` and reports test accuracy on random samples.

In the saved notebook run, validation accuracy reached about **0.76** after 10 epochs and test accuracy was about
**0.59**.

## Files

| File | What it is |
|------|------------|
| `image_prediction_CIFAR.ipynb` | The notebook, with explanations and outputs |
| `image_prediction_CIFAR.py` | The same pipeline as a script |

## Run it

The code uses the TensorFlow 1.x graph API (`tf.placeholder`, `tf.Session`), so it needs TensorFlow 1.x, or 2.x with
`tf.compat.v1` and `tf.disable_v2_behavior()`.

```bash
pip install "tensorflow<2" numpy matplotlib scikit-learn tqdm
python image_prediction_CIFAR.py
```

Or open the notebook with `jupyter notebook`.

## Licence

MIT, see [LICENSE.md](LICENSE.md).
