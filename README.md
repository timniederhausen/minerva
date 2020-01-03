# minerva

Library to perform dimensionality reduction via autoencoders.

The autoencoder topology to use is chosen based on reliable auto-encoders of already known datasets, which are adapted to the dataset in question.
This metalearning step relies on the dataset's metafeatures to establish a notion of similarity.

A selection of simple training datasets are part of this library.

### Example

```py
import numpy as np

from minverva import autoencoder_for_dataset

# exposition only
from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

autoencoder = autoencoder_for_dataset(x_train)
x_train_encoded = autoencoder.encode(x_train)
x_test_encoded = autoencoder.encode(x_test)

# Now you can perform classification using |x_train_encoded| and |x_test_encoded|
# ...
```
