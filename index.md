## TensorFlow + tflean

Launching Tensorboard
```bash
$ tensorboard --logdir=<logdir> --port=<port>
``` 

from numpy to tensor
```python
1. tf.convert_to_tensor(np.array)
2. tf.placeholder
```

Fine Tuning
restore=False
```python
softmax = fully_connected(network, num_classes, restore=False)
```

---
## TensorFlow(Gpu Ver.) Installation @160507

1) cuda 7.5
```bash
$ sudo apt-get install nvidia-361
$ sudo apt-get install nvidia-cuds-toolkit
$ sudo reboot
```

2) cuDNN 4
```bash
# download .whl @firefox (directly access the follwing link)
# https://developer.nvidia.com/rdp/form/cudnn-download-survey
$ tar xzf cudnn-7.0-linux-x64-v4.0-prod.tgz
$ sudo cp -a cuda/lib64/* /usr/local/lib/
$ sudo cp -a cuda/include/* /usr/local/include/
$ sudo ldconfig
```

3)tensorflow
```bash
$ sudo apt-get install python-dev python-virtualenv
# download .whl @firefox (directly access the follwing link)
# https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl
$ cd Download
$ sudo pip install tensorflow-0.8.0-cp27-none-linux_x86_64.whl
```

reference
> http://qiita.com/yukoba/items/3692f1cb677b2383c983
> http://qiita.com/akiraak/items/1c7fb452fb7721292071#cudnn-v4-%E3%81%AE%E3%82%A4%E3%83%B3%E3%82%B9%E3%83%88%E3%83%BC%E3%83%AB
> https://www.tensorflow.org/versions/master/get_started/os_setup.html

---
## Theory

preprocessing
- to center the data to have mean of zero, and normalize its scale to [-1, 1] along each feature

initialization
- xavier initialization
- w = np.random.randn(n) * sqrt(2.0/n)
- identity matrix initialization for RNN

regularization
- l2 regularization with cross validation
- max norm constraint(gradient clipping for RNN)
- dropout 0.5
- use batch normalization

small dateset
- pretraining
- data distortion to increse the number of datas
- not split dataset, but use cross-validation

model construction flow
1. compare train vs valid accuracy
2. if not overfit: make more complicated model
3. if overfit: regularize

loss
- l2loss is hard to optimize! don’t use dropout!!
- softmax is better

### optimizer

sgd

momentum
* A typical setting is to start with momentum of about 0.5 and anneal it to 0.99 or so over multiple epochs.

Nesterov Momentum

adagrad
```python
# Assume the gradient dx and parameter vector x
cache += dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

RMSprop
```python
cache = decay_rate * cache + (1 - decay_rate) * dx**2
x += - learning_rate * dx / (np.sqrt(cache) + eps)
```

Adam
looks a bit like RMSProp with momentum
```python
m = beta1*m + (1-beta1)*dx
v = beta2*v + (1-beta2)*(dx**2)
x += - learning_rate * m / (np.sqrt(v) + eps)
```

## others
model ensembles

annealing the learning rate

set learning rate to realize that update_scale / param_scale = ~1e-3 , when using convnet, the first layer's weight
- step decay
- exponential decay
- 1/t decay

large number of target class
- negative sampling
- sampled softmax