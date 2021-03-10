"""
Bug submitted at
https://github.com/keras-team/autokeras/issues/1528

"""

import tensorflow as tf
import numpy as np
import autokeras as ak
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])


class Gen:

    def __init__(self, train):
        self.used_once = False
        if train:
            self.x = x_train
            self.y = y_train
            self.name = "train_dataset"
        else:
            self.x = x_test
            self.y = y_test
            self.name = "test_dataset"

    def gen(self):
        print(" I am Groot ", self.name)

        if self.used_once:
            print(
                f"Happened twice in {self.name} ... "
                f"groot is already confusing ... will raise error")
            raise Exception(
                f"was already used ... exception raised in {self.name}")
        else:
            self.used_once = True
            for i, _ in enumerate(zip(x_train, y_train)):
                if i == 3:
                    return
                yield _


def behaves_as_expected():
    ds_train = tf.data.Dataset.from_generator(
        generator=Gen(train=True).gen,
        output_signature=(
            tf.TensorSpec(shape=(28, 28), dtype=np.int32),
            tf.TensorSpec(shape=(), dtype=np.uint8)
        )
    ).batch(10)

    ds_validation = tf.data.Dataset.from_generator(
        generator=Gen(train=False).gen,
        output_signature=(
            tf.TensorSpec(shape=(28, 28), dtype=np.int32),
            tf.TensorSpec(shape=(), dtype=np.uint8)
        )
    ).batch(10)

    model.fit(x=ds_train, validation_data=ds_validation)


def does_not_behave_as_expected():
    ds_train = tf.data.Dataset.from_generator(
        generator=Gen(train=True).gen,
        output_signature=(
            tf.TensorSpec(shape=(28, 28), dtype=np.int32),
            tf.TensorSpec(shape=(), dtype=np.uint8)
        )
    ).batch(10)

    ds_validation = tf.data.Dataset.from_generator(
        generator=Gen(train=False).gen,
        output_signature=(
            tf.TensorSpec(shape=(28, 28), dtype=np.int32),
            tf.TensorSpec(shape=(), dtype=np.uint8)
        )
    ).batch(10)

    clf = ak.ImageClassifier(
        overwrite=True,
        max_trials=100
    )

    # Feed the tensorflow Dataset to the classifier.
    # noinspection PyTypeChecker
    clf.fit(
        x=ds_train,
        validation_data=ds_validation,
        epochs=10
    )


if __name__ == '__main__':

    behaves_as_expected()
    does_not_behave_as_expected()


"""
The output is
```
 I am Groot  train_dataset
      1/Unknown - 1s 790ms/step - loss: 2.2792 - accuracy: 0.3333 I am Groot  test_dataset
1/1 [==============================] - 1s 1s/step - loss: 2.2792 - accuracy: 0.3333 - val_loss: 2.2683 - val_accuracy: 0.0000e+00
 I am Groot  train_dataset
 I am Groot  train_dataset
Happened twice in train_dataset ... groot is already confusing ... will raise error
```

This is followed by a raised exception in generator ...
"""