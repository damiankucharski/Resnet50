# Implementation of Resnet50 architecture

- [Implementation of Resnet50 architecture](#implementation-of-resnet50-architecture)
  - [Architecure](#architecure)
  - [Usage](#usage)


</br>

It is based on [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) paper and on what I've learned in [this coursera course](https://www.coursera.org/learn/convolutional-neural-networks?).

## Architecure

</br>

The model is built using this architecture provided in aforementioned paper:
</br>
</br>

![Architecure](scheme.jpg)

</br>

## Usage
</br>

To use the model you can just import it, compile and fit. Initializer will build architecture auotmatically:

```Python
from net import Resnet50

#...

net = Resnet50()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
net.fit(X,y)
results = net.evaluate(X_test, y_test)
print(results)

#...

```

