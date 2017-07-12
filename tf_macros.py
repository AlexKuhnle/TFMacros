from itertools import chain, combinations
from math import sqrt
import tensorflow as tf


def rank(x):
    assert isinstance(x, tf.Tensor)
    return x.shape.ndims


def shape(x):
    assert isinstance(x, tf.Tensor)
    return tuple(-1 if dims.value is None else dims.value for dims in x.shape.dims)


def product(xs):
    prod = 1
    for x in xs:
        prod *= x
    return prod


def make_least_common_shape(xs):
    common_rank = rank(xs[0])
    assert all(rank(x) == common_rank for x in xs)
    shapes = [shape(x) for x in xs]
    least_common_shape = tuple(max(s[r] for s in shapes) for r in range(common_rank))
    ys = list()
    for x, s in zip(xs, shapes):
        if all(s[r] == least_common_shape[r] for r in range(common_rank)):
            continue
        x = tf.tile(input=x, multiples=[common_dims if dims == 1 else 1 for dims, common_dims in zip(s, least_common_shape)])
        assert shape(x) == least_common_shape
        ys.append(x)
    return ys


def make_broadcastable(xs):
    broadcastable_shape = shape(max(xs, key=lambda x: rank(x) - 0.5 * sum(dims == 1 for dims in shape(x))))
    ys = list()
    for x in xs:
        x_shape = shape(x)
        if len(x_shape) == len(broadcastable_shape):
            assert all(dims == ref_dims or dims == 1 for dims, ref_dims in zip(x_shape, broadcastable_shape))
            ys.append(x)
            continue
        r = len(x_shape) - 1
        last_dims = None
        for dims in reversed(broadcastable_shape):
            if r >= 0 and x_shape[r] == dims:
                r -= 1
                last_dims = dims
            else:
                print(x_shape, broadcastable_shape)
                assert dims != last_dims
                x = tf.expand_dims(input=x, axis=(r + 1))
        assert r == -1
        ys.append(x)
    return ys


class Model(object):

    current = None

    precision = 32
    tensors = dict()
    variables = dict()
    placeholders = dict()
    variable_keys = []

    @staticmethod
    def dtype(dtype):
        assert dtype in ('float', 'int')
        return dtype + str(Model.precision)

    @staticmethod
    def add_tensor(name, tensor):
        assert name not in Model.tensors
        Model.tensors[name] = tensor

    def __init__(self, optimizer='adam', learning_rate=0.001, name=None, model_file=None):
        assert optimizer in ('adam',)
        assert isinstance(learning_rate, float)
        assert name is None or isinstance(name, str)
        assert model_file is None or isinstance(model_file, str)
        self.model_file = model_file
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.name = name
        self.session = None
        self.defined = False

    def __str__(self):
        return 'Model({})'.format(self.name)

    def __enter__(self):
        assert Model.current is None
        Model.current = self
        self.scope = tf.name_scope(self.name)
        self.scope.__enter__()
        Input(name='dropout', shape=(), batched=False).forward()
        self.dropout = Model.placeholders.pop('dropout')
        return self

    def __exit__(self, type, value, tb):
        if type is not None:
            assert False
        assert Model.current is not None
        Model.current = None
        if self.defined:
            self.coordinator.request_stop()
            self.coordinator.join(threads=self.queue_threads)
            if self.model_file:
                self.saver.save(sess=self.session, save_path=self.model_file)
        else:
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            loss = tf.losses.get_total_loss()
            Model.add_tensor(name='loss', tensor=loss)
            grads_and_vars = optimizer.compute_gradients(loss=loss)
            self.optimization = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            self.scope.__exit__(type, value, tb)

    def finalize(self, restore=False):
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        loss = tf.losses.get_total_loss()
        Model.add_tensor(name='loss', tensor=loss)
        grads_and_vars = optimizer.compute_gradients(loss=loss)
        self.optimization = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        self.scope.__exit__(None, None, None)
        self.defined = True

        self.session = tf.Session()
        if self.model_file:
            self.saver = tf.train.Saver()
        if restore:
            assert self.model_file
            self.saver.restore(sess=self.session, save_path=self.model_file)
        else:
            self.session.run(tf.global_variables_initializer())
        self.coordinator = tf.train.Coordinator()
        self.queue_threads = tf.train.start_queue_runners(sess=self.session, coord=self.coordinator)

    def __call__(self, query=None, data=None, optimize=True, dropout=0.0):
        assert self.session
        if query is None:
            fetches = dict()
        elif isinstance(query, str):
            fetches = dict(query=Model.tensors[query])
        else:
            fetches = {name: Model.tensors[name] for name in query}
        if optimize:
            assert 'optimization' not in fetches
            fetches['optimization'] = self.optimization
        if data is None:
            feed_dict = None
        elif isinstance(data, dict):
            feed_dict = {Model.placeholders[value]: data[value] for value in Model.placeholders}
        else:
            assert len(Model.placeholders) == 1
            feed_dict = {next(iter(Model.placeholders.values())): data}
        if dropout > 0.0:
            assert dropout < 1.0
            feed_dict[self.dropout] = dropout
        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)
        if optimize:
            fetched.pop('optimization')
        return fetched


class Unit(object):

    num_in = -1
    num_out = -1

    def __init__(self, name=None, variable_key=None):
        name = name if name is None or isinstance(name, str) else tuple(name)
        assert name is None or isinstance(name, str) or isinstance(name, tuple)
        assert variable_key is None or isinstance(variable_key, str)
        self.name = name
        self.variable_key = variable_key
        self.output = None

    def __str__(self):
        return self.__class__.__name__

    def subkey(self, key):
        return key if self.variable_key else None

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        if self.output is None:
            with tf.name_scope(str(self)):
                if self.variable_key:
                    Model.variable_keys.append(self.variable_key)
                output = self.forward(*inputs)
                if isinstance(output, tf.Tensor):
                    self.output = output
                    if self.name is not None:
                        assert isinstance(self.name, str)
                        Model.add_tensor(name=self.name, tensor=self.output)
                elif len(output) == 1:
                    self.output = output[0]
                    if self.name is not None:
                        assert isinstance(self.name, str)
                        Model.add_tensor(name=self.name, tensor=self.output)
                else:
                    self.output = tuple(output)
                    if self.name is not None:
                        assert isinstance(self.name, tuple)
                        for name, output in zip(self.name, self.output):
                            Model.add_tensor(name=name, tensor=output)
                if self.variable_key:
                    Model.variable_keys.pop()
        return self.output

    def __rshift__(self, other):
        assert isinstance(other, Unit)
        inputs = self()
        inputs = (inputs,) if isinstance(inputs, tf.Tensor) else inputs
        return other(*inputs)

    def __rrshift__(self, other):
        composed = False
        if isinstance(other, tf.Tensor):
            inputs = (other,)
        else:
            inputs = []
            for x in other:
                if isinstance(x, tf.Tensor):
                    inputs.append(x)
                elif isinstance(x, Composed):
                    inputs.append(x)
                    composed = True
                else:
                    x = x()
                    assert isinstance(x, tf.Tensor)
                    inputs.append(x)
        if composed:
            return Composed(first=inputs, second=self)
        else:
            return self(*inputs)


class Composed(Unit):

    def __init__(self, first, second):
        super(Composed, self).__init__()
        self.first = first
        self.second = second

    def forward(self, *xs):
        return xs >> self.first >> self.second

    def __rshift__(self, other):
        assert isinstance(other, Unit)
        return Composed(first=(self,), second=other)


class Layer(Unit):

    num_in = 1
    num_out = 1

    def __init__(self, size, name=None, variable_key=None):
        assert self.__class__.num_in == 1 and self.__class__.num_out == 1
        super(Layer, self).__init__(variable_key=variable_key)
        assert isinstance(size, int) and size > 0
        self.size = size


class LayerStack(Unit):

    def __init__(self, name=None, variable_key=None):
        super(LayerStack, self).__init__(variable_key=variable_key)
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x >>= layer
        return x


class Variable(Unit):

    num_in = 0
    num_out = 1

    def __init__(self, key, shape, dtype='float', init='out', value=None, name=None):
        assert self.__class__.num_in == 0 and self.__class__.num_out == 1
        super(Variable, self).__init__(name=name, variable_key=key)
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        assert len(shape) > 0 and all(isinstance(n, int) and n > 0 for n in shape)
        assert init in ('constant', 'zeros', 'ones', 'in', 'out', 'in-out', 'stddev') or Nonlinearity.valid(init)
        assert init in ('constant', 'zeros', 'ones') or dtype == 'float'
        self.shape = shape
        self.dtype = Model.dtype(dtype=dtype)
        self.init = init
        self.value = value

    def forward(self):
        if self.init == 'constant':
            assert self.value is not None
            initializer = self.value
        elif self.init == 'zeros':
            initializer = tf.zeros(shape=self.shape, dtype=self.dtype)
        elif self.init == 'ones':
            initializer = tf.ones(shape=self.shape, dtype=self.dtype)
        else:
            if self.init == 'stddev':
                assert self.value is not None
                stddev = self.value
            elif self.init == 'selu':
                stddev = sqrt(1.0 / self.shape[-1])
            elif self.init == 'out':
                stddev = sqrt(2.0 / self.shape[-1])
            elif self.init == 'in' or self.init in ('elu', 'relu'):
                assert len(self.shape) >= 2
                stddev = sqrt(2.0 / self.shape[-2])
            elif self.init == 'in-out' or Nonlinearity.valid(self.init):
                assert len(self.shape) >= 2
                stddev = sqrt(2.0 / (self.shape[-2] + self.shape[-1]))
            initializer = tf.truncated_normal(shape=self.shape, stddev=stddev)
        if Model.variable_keys:
            key = '/'.join(Model.variable_keys) + '/' + self.variable_key
            if key not in Model.variables:
                Model.variables[key] = variable = tf.Variable(initial_value=initializer, trainable=(self.init != 'constant'), name=key, dtype=self.dtype)
            else:
                variable = Model.variables[key]
        else:
            variable = tf.Variable(initial_value=initializer, trainable=(self.init != 'constant'), name=self.variable_key, dtype=self.dtype)
        return tf.identity(input=variable)


class Input(Unit):

    num_in = 0
    num_out = 1

    def __init__(self, name, shape, dtype='float', batched=True, tensor=None, variable_key=None):
        assert isinstance(name, str)
        super(Input, self).__init__(name=name, variable_key=variable_key)
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        assert all(isinstance(n, int) and (n > 0 or n == -1) for n in shape)
        assert isinstance(batched, bool)
        self.shape = tuple(None if x == -1 else x for x in shape)
        self.dtype = Model.dtype(dtype=dtype)
        if batched:
            self.shape = (None,) + self.shape
        self.tensor = tensor

    def forward(self):
        if self.tensor is None:
            assert self.name not in Model.placeholders
            Model.placeholders[self.name] = placeholder = tf.placeholder(dtype=self.dtype, shape=self.shape)
            return tf.identity(input=placeholder)
        else:
            return tf.identity(input=self.tensor)


class Binary(Input):

    num_in = 1

    def __init__(self, name, soft=0.0, tensor=None, variable_key=None):
        super(Binary, self).__init__(name=name, shape=(), batched=True, tensor=tensor, variable_key=variable_key)
        assert isinstance(soft, float) and 0.0 <= soft < 0.5
        self.soft = soft

    def forward(self, x):
        correct = y = super(Binary, self).forward()
        if self.soft > 0.0:
            noise = tf.random_uniform(shape=(1, 1), minval=0.0, maxval=self.soft)
            correct = tf.abs(x=(correct - noise))
        x >>= Linear(size=1, variable_key=self.subkey('prediction'))
        x = (tf.tanh(x=x) + 1.0) / 2.0
        cross_entropy = -(correct * tf.log(x=x + 1e-10) + (1.0 - correct) * tf.log(x=1.0 - x + 1e-10))
        loss = tf.reduce_mean(input_tensor=cross_entropy)
        tf.losses.add_loss(loss=loss)
        prediction = tf.cast(x=tf.greater(x=x, y=tf.constant(value=0.5)), dtype=Model.dtype('float'))
        correct = tf.cast(x=tf.equal(x=prediction, y=correct), dtype=Model.dtype('float'))
        accuracy = tf.reduce_mean(input_tensor=correct)
        Model.add_tensor(name=self.name + '-accuracy', tensor=accuracy)
        return y


class Classification(Input):

    def __init__(self, name, num_classes, multi_class=False, soft=0.0, tensor=None, variable_key=None):
        super(Classification, self).__init__(name=name, shape=(num_classes,), batched=True, tensor=tensor, variable_key=variable_key)
        assert isinstance(num_classes, int) and num_classes > 0
        assert isinstance(multi_class, bool)
        assert isinstance(soft, float) and 0.0 <= soft < 0.5
        self.num_classes = num_classes
        self.multi_class = multi_class
        self.soft = soft

    def forward(self, x):
        correct = y = super(Classification, self).forward()
        if not self.multi_class and rank(correct) == 1:
            correct = tf.one_hot(indices=correct, depth=self.num_classes)
        if self.soft > 0.0:
            noise = tf.random_uniform(shape=(1, shape(correct)[1]), minval=0.0, maxval=self.soft)
            correct = tf.abs(x=(correct - noise))
        x >>= Linear(size=self.num_classes, variable_key=self.subkey('prediction'))
        if self.multi_class:
            tf.losses.sigmoid_cross_entropy(multi_class_labels=correct, logits=x)
        else:
            tf.losses.softmax_cross_entropy(onehot_labels=correct, logits=x)
        prediction = tf.one_hot(indices=tf.argmax(input=x, axis=1), depth=self.num_classes)
        relevant = tf.reduce_sum(input_tensor=correct, axis=1)
        selected = tf.reduce_sum(input_tensor=prediction, axis=1)
        true_positive = tf.reduce_sum(input_tensor=tf.minimum(x=prediction, y=correct), axis=1)
        precision = tf.reduce_mean(input_tensor=tf.divide(x=true_positive, y=selected), axis=0)
        recall = tf.reduce_mean(input_tensor=tf.divide(x=true_positive, y=relevant), axis=0)
        Model.add_tensor(name=self.name + '-precision', tensor=precision)
        Model.add_tensor(name=self.name + '-recall', tensor=recall)
        return y


class Distance(Input):

    def __init__(self, name, shape, tensor=None):
        super(Distance, self).__init__(name=name, shape=shape, batched=True, tensor=tensor)

    def forward(self, x):
        correct = y = super(Distance, self).forward()
        tf.losses.mean_squared_error(labels=x, predictions=correct)
        return y


class Identity(Unit):

    num_in = 1
    num_out = 1

    def __init__(self, name=None):
        super(Identity, self).__init__(name=name)

    def forward(self, *xs):
        assert len(xs) >= 1
        return xs[0] if len(xs) == 1 else xs


class Print(Unit):

    def __init__(self, size=10, times=None, name=None):
        super(Print, self).__init__(name=name)
        assert isinstance(size, int) and size > 0
        assert times is None or isinstance(times, int) and times > 0
        self.size = size
        self.times = times

    def forward(self, *xs):
        return (tf.Print(input_=xs[0], data=xs, summarize=self.size, first_n=self.times),) + tuple(xs[1:])


class Select(Unit):

    def __init__(self, index, name=None):
        super(Select, self).__init__(name=name)
        assert isinstance(index, int) and index >= 0
        self.index = index

    def forward(self, *xs):
        assert len(xs) > self.index
        return xs[self.index]


class Nonlinearity(Unit):

    @staticmethod
    def valid(nonlinearity):
        return nonlinearity in ('elu', 'relu', 'sigmoid', 'softmax', 'tanh')

    def __init__(self, nonlinearity='relu', name=None):
        super(Nonlinearity, self).__init__(name=name)
        assert Nonlinearity.valid(nonlinearity)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        if self.nonlinearity == 'elu':
            return tf.nn.elu(features=x)
        elif self.nonlinearity == 'relu':
            return tf.nn.relu(features=x)
        elif self.nonlinearity == 'selu':
            # https://arxiv.org/pdf/1706.02515.pdf
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.where(condition=(x >= 0.0), x=x, y=(alpha * tf.nn.elu(features=x)))
        elif self.nonlinearity == 'sigmoid':
            return tf.sigmoid(x=x)
        elif self.nonlinearity == 'softmax':
            return tf.nn.softmax(logits=x)
        elif self.nonlinearity == 'tanh':
            return tf.nn.tanh(x=x)


class Dropout(Unit):

    num_in = 1
    num_out = 1

    def __init__(self, name=None):  # potentially separate dropout
        super(Dropout, self).__init__(name=name)

    def forward(self, x):
        assert Model.current.dropout
        return tf.nn.dropout(x=x, keep_prob=(1.0 - Model.current.dropout))


class Normalization(Unit):

    num_in = 1
    num_out = 1

    def __init__(self, offset=False, scale=False, variance_epsilon=1e-6, name=None, variable_key=None):
        super(Normalization, self).__init__(name=name, variable_key=variable_key)
        assert isinstance(offset, bool)
        assert isinstance(scale, bool)
        assert isinstance(variance_epsilon, float) and variance_epsilon > 0.0
        self.offset = offset
        self.scale = scale
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        mean, variance = tf.nn.moments(x=x, axes=tuple(range(rank(x) - 1)))
        self.offset = Variable(key=self.subkey('offset'), shape=shape(mean), init='zeros') if self.offset else None
        self.scale = Variable(key=self.subkey('scale'), shape=shape(mean), init='zeros') if self.scale else None
        return tf.nn.batch_normalization(x=x, mean=mean, variance=variance, offset=(self.offset() if self.offset else None), scale=(self.scale() if self.scale else None), variance_epsilon=self.variance_epsilon)


class Reduction(Unit):

    num_in = -1
    num_out = 1

    requires_uniform_shape = ('collapse', 'conv', 'max', 'mean', 'prod', 'sum')

    @staticmethod
    def valid(reduction):
        return isinstance(reduction, str) and reduction in ('cbp', 'collapse', 'concat', 'conv', 'conv2d', 'last', 'max', 'mean', 'prod', 'stack', 'sum')

    def __init__(self, reduction='avg', axis=-1, arg=None, name=None, variable_key=None):
        super(Reduction, self).__init__(name=name, variable_key=variable_key)
        axis = (axis, axis) if isinstance(axis, int) else tuple(axis)
        assert Reduction.valid(reduction)
        assert len(axis) == 2 and isinstance(axis[0], int) and isinstance(axis[1], int)
        self.reduction = reduction
        self.axis = axis
        if self.reduction == 'cbp':
            if arg is None:
                self.reduction_layer = CompactBilinearPooling(variable_key=self.subkey('cbp'))
            else:
                self.reduction_layer = CompactBilinearPooling(size=arg, variable_key=self.subkey('cbp'))
        elif self.reduction == 'conv':
            self.reduction_layer = Convolution(size=1, window=1, normalization=False, nonlinearity='relu', variable_key=self.subkey('convolution'))
        elif self.reduction == 'concat':
            if arg is None:
                self.reduction_layer = Concatenation()
            else:
                self.reduction_layer = Concatenation(axis=arg)
        elif self.reduction == 'stack':
            if arg is None:
                self.reduction_layer = Stack()
            else:
                self.reduction_layer = Stack(axis=arg)
        else:
            assert arg is None
            self.reduction_layer = None

    def forward(self, *xs):
        multiple_inputs = len(xs) > 1
        if multiple_inputs:
            assert self.axis[0] == self.axis[1]
            # xs = [x if shape(x) == shape(xs[0]) else tf.expand_dims(input=x, axis=self.axis[0]) for x in xs]
            xs = make_broadcastable(xs)
            assert all(rank(x) == rank(xs[0]) for x in xs)
            axis = self.axis[0] if self.axis[0] >= 0 else rank(xs[0]) + self.axis[0] + 1
            if self.reduction == 'max':
                y = xs[0]
                for x in xs[1:]:
                    y = tf.maximum(x=y, y=x)
                return y
            elif self.reduction == 'mean':
                y = xs[0]
                for x in xs[1:]:
                    y = tf.add(x=y, y=x)
                return y / float(len(xs))
            elif self.reduction == 'min':
                y = xs[0]
                for x in xs[1:]:
                    y = tf.minimum(x=y, y=x)
                return y
            elif self.reduction == 'prod':
                y = xs[0]
                for x in xs[1:]:
                    y = tf.multiply(x=y, y=x)
                return y
            elif self.reduction == 'sum':
                y = xs[0]
                for x in xs[1:]:
                    y = tf.add(x=y, y=x)
                return y
            elif self.reduction in Reduction.requires_uniform_shape:
                xs = make_least_common_shape(xs)
                x = tf.stack(values=xs, axis=axis)
            else:
                x = xs[0]
        else:
            x = xs[0]
        start = self.axis[0] if self.axis[0] >= 0 else rank(x) + self.axis[0]
        end = self.axis[1] + 1 if self.axis[1] >= 0 else rank(x) + self.axis[1] + 1
        assert rank(x) > max(start, end - 1)
        if start >= end:
            return x
        elif self.reduction == 'collapse':
            reduced_shape = shape(x)
            reduced_shape = reduced_shape[:start] + (product(reduced_shape[start:end]),) + reduced_shape[end:]
            return tf.reshape(tensor=x, shape=reduced_shape)
        elif self.reduction == 'conv':
            if multiple_inputs:
                assert start + 1 == end == rank(x)
            else:
                assert start == 1
                assert end == rank(x) - 1
                reduced_shape = shape(x)
                reduced_shape = reduced_shape[:start] + (product(reduced_shape[start:end]),) + reduced_shape[end:]
                x = tf.transpose(a=tf.reshape(tensor=x, shape=reduced_shape), perm=(0, 2, 1))
            x >>= self.reduction_layer
            return tf.squeeze(input=x, axis=2)
        elif self.reduction == 'conv2d':
            assert start == 1 and end == 3
            self.reduction_layer = Convolution(size=shape(x)[-1], window=(shape(x)[1], shape(x)[2]), normalization=False, nonlinearity='relu', padding='VALID', variable_key=self.subkey('convolution'))
            x >>= self.reduction_layer
            return tf.squeeze(input=tf.squeeze(input=x, axis=2), axis=1)
        elif self.reduction == 'last':
            if multiple_inputs:
                return xs[-1]
            else:
                for axis in reversed(range(start, end)):
                    if axis == 0:
                        x = x[-1, ...]
                    elif axis == 1:
                        x = x[:, -1, ...]
                    elif axis == 2:
                        x = x[:, :, -1, ...]
                    elif axis == 3:
                        x = x[:, :, :, -1, ...]
                    elif axis == 4:
                        x = x[:, :, :, :, -1, ...]
                    else:
                        assert False
                return x
        elif self.reduction == 'max':  # many axes at once?
            for axis in reversed(range(start, end)):
                x = tf.reduce_max(input_tensor=x, axis=axis)
            return x
        elif self.reduction == 'mean':
            for axis in reversed(range(start, end)):
                x = tf.reduce_mean(input_tensor=x, axis=axis)
            return x
        elif self.reduction == 'prod':
            for axis in reversed(range(start, end)):
                x = tf.reduce_prod(input_tensor=x, axis=axis)
            return x
        elif self.reduction == 'sum':
            for axis in reversed(range(start, end)):
                x = tf.reduce_sum(input_tensor=x, axis=axis)
            return x
        else:
            assert self.reduction_layer
            # more general since accepts different shapes !!!
            if not multiple_inputs:
                xs = [x]
                for axis in reversed(range(start, end)):
                    xs = [y for x in xs for y in tf.unstack(value=x, axis=axis)]
            return xs >> self.reduction_layer


class Concatenation(Unit):

    def __init__(self, axis=-1, name=None):
        super(Concatenation, self).__init__(name=name)
        assert isinstance(axis, int)
        self.axis = axis

    def forward(self, *xs):
        assert len(xs) >= 1 and all(rank(x) == rank(xs[0]) for x in xs)  # what if length 0?
        axis = self.axis if self.axis >= 0 else rank(xs[0]) + self.axis
        assert rank(xs[0]) > axis
        return tf.concat(values=xs, axis=axis)


class Stack(Unit):

    def __init__(self, axis=-1, name=None):
        super(Stack, self).__init__(name=name)
        assert isinstance(axis, int)
        self.axis = axis

    def forward(self, *xs):
        assert len(xs) >= 1 and all(rank(x) == rank(xs[0]) for x in xs)
        axis = self.axis if self.axis >= 0 else rank(xs[0]) + self.axis + 1
        assert rank(xs[0]) >= axis
        return tf.stack(values=xs, axis=axis)


class CompactBilinearPooling(Unit):

    def __init__(self, size=None, name=None, variable_key=None):
        super(CompactBilinearPooling, self).__init__(name=name, variable_key=variable_key)
        # assert GPU !!!
        # default arg size
        assert size is None or (isinstance(size, int) and size > 0)
        self.size = size

    def forward(self, *xs):
        assert len(xs) >= 1 and all(rank(x) == rank(xs[0]) for x in xs)
        # what if length 0?
        size = shape(xs[0])[-1] if self.size is None else self.size
        p = None
        for n, x in enumerate(xs):
            input_size = shape(x)[-1]
            indices = tf.range(start=input_size, dtype=tf.int64)
            indices = tf.expand_dims(input=indices, axis=1)
            sketch_indices = tf.random_uniform(shape=(input_size,), maxval=size, dtype=tf.int64)
            self.sketch_indices = Variable(key=self.subkey('indices' + str(n)), shape=input_size, init='constant', value=sketch_indices)
            sketch_indices = tf.expand_dims(input=self.sketch_indices(), axis=1)
            sketch_indices = tf.concat(values=(indices, sketch_indices), axis=1)
            sketch_values = tf.random_uniform(shape=(input_size,))
            self.sketch_values = Variable(key=self.subkey('values' + str(n)), shape=input_size, init='constant', value=sketch_values)
            sketch_values = tf.round(x=self.sketch_values())
            sketch_values = sketch_values * 2 - 1
            sketch_matrix = tf.SparseTensor(indices=sketch_indices, values=sketch_values, dense_shape=(input_size, size))
            sketch_matrix = tf.sparse_reorder(sp_input=sketch_matrix)
            x = tf.sparse_tensor_dense_matmul(sp_a=sketch_matrix, b=x, adjoint_a=True, adjoint_b=True)
            x = tf.transpose(a=x)
            x = tf.reshape(tensor=x, shape=(shape(x)[0] or -1, size))
            x = tf.complex(real=x, imag=0.0)
            x = tf.fft(input=x)
            if p is None:
                p = x
            else:
                x = make_broadcastable(x=x, reference=p)
                p = tf.multiply(x=p, y=x)
        return tf.ifft(input=tf.real(input=p))


class Pooling(Unit):

    @staticmethod
    def valid(pool):
        return isinstance(pool, str) and pool in ('none', 'average', 'avg', 'max', 'maximum')

    def __init__(self, pool='max', window=(2, 2), stride=2, padding='SAME', name=None):
        super(Pooling, self).__init__(name=name)
        window = tuple(window)
        assert Pooling.valid(pool)
        assert len(window) == 2 and all(isinstance(n, int) and n > 0 for n in window)
        assert padding in ('SAME', 'VALID')
        self.pool = pool
        self.window = (1, window[0], window[1], 1)
        self.padding = padding
        if isinstance(stride, int):
            assert stride > 0
            self.stride = stride if len(window) == 1 else (1, stride, stride, 1)
        else:
            assert len(stride) == 2 and stride[0] > 0 and stride[1] > 0
            self.stride = (1, stride[0], stride[1], 1)

    def forward(self, x):
        if self.pool == 'none':
            return x
        elif self.pool in ('avg', 'average'):
            return tf.nn.avg_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)
        elif self.pool in ('max', 'maximum'):
            return tf.nn.max_pool(value=x, ksize=self.window, strides=self.stride, padding=self.padding)


    # def unpool(self, x, unpooling_type='zero'):  # zero, id
    #     assert NeuralNetwork.rank(x) == 4 and NeuralNetwork.shape(x)[0] is None
    #     width, height, size = NeuralNetwork.shape(x)[1:]
    #     with tf.name_scope('unpool'):
    #         if unpooling_type == 'zero':
    #             zeros = tf.zeros_like(tensor=x)
    #             x = tf.stack(values=(x, zeros, zeros, zeros), axis=4)
    #             x = tf.reshape(tensor=x, shape=(-1, width, height, size, 2, 2))
    #             # x = tf.Print(x, (x[0,0,0,:,0,0],x[0,0,0,:,0,1],x[0,0,0,:,1,0],x[0,0,0,:,1,1]))
    #             x = tf.transpose(a=x, perm=(0, 1, 5, 2, 4, 3))
    #         elif unpooling_type == 'id':
    #             # x = tf.stack(values=(x, x, x, x), axis=4)
    #             x = tf.tile(input=x, multiples=(1, 1, 1, 4))
    #             x = tf.reshape(tensor=x, shape=(-1, width, height, 2, 2, size))
    #             # x = tf.Print(x, (x[0,0,0,0,0,:],x[0,0,0,0,1,:],x[0,0,0,1,0,:],x[0,0,0,1,1,:]))
    #             x = tf.transpose(a=x, perm=(0, 1, 4, 2, 3, 5))
    #         x = tf.reshape(tensor=x, shape=(-1, width * 2, height * 2, size))
    #     # x = tf.Print(x, (tf.reduce_all([tf.reduce_all([x[n,2*w,2*h,:] == y[n,w,h,:], x[n,2*w,2*h+1,:] == 0, x[n,2*w+1,2*h,:] == 0, x[n,2*w+1,2*h+1,:] == 0]) for n in range(128) for w in range(width) for h in range(height)]),))
    #     # x = tf.Print(x, (x[0,0,0,:],x[0,0,1,:],x[0,1,0,:],x[0,1,1,:]))
    #     # x = tf.Print(x, (x[0,2,0,:],x[0,2,1,:],x[0,3,0,:],x[0,3,1,:]))
    #     # x = tf.Print(x, (x[0,-2,-2,:],x[0,-2,-1,:],x[0,-1,-2,:],x[0,-1,-1,:]))
    #     assert NeuralNetwork.shape(x) == [None, width * 2, height * 2, size]
    #     return x


class Embedding(Unit):

    def __init__(self, indices, size, name=None, variable_key=None):
        super(Embedding, self).__init__(name=name, variable_key=variable_key)
        assert isinstance(indices, int) and indices > 0
        assert isinstance(size, int) and size > 0
        self.embeddings = Variable(key=self.subkey('embeddings'), shape=(indices, size))

    def forward(self, x):
        return tf.nn.embedding_lookup(params=self.embeddings(), ids=x)


class Split(Unit):

    def __init__(self, axis=1, size=1, reduction=None, name=None):
        super(Split, self).__init__(name=name)
        axis = (axis,) if isinstance(axis, int) else tuple(sorted(axis, reverse=True))
        size = (size,) if isinstance(size, int) else tuple(size)
        assert all(isinstance(a, int) and a >= 0 for a in axis)
        assert all(isinstance(s, int) and s > 0 for s in size)
        self.axis = axis
        self.size = size
        self.reduction = None if reduction is None else Reduction(reduction=reduction)

    def forward(self, x):
        xs = [x]
        for a in self.axis:
            xs = [y for ys in xs for y in tf.unstack(value=ys, axis=a)]
        if self.size != (1,):
            xs = chain(*(combinations(xs, r=s) for s in self.size))  # others interesting? permutation, product?
        if self.reduction is not None:
            xs = [y >> self.reduction for y in xs]
        return tuple(xs)


class Relational(Unit):

    def __init__(self, relation_unit, axis=1, relation_reduction='concat', reduction='sum', name=None, variable_key=None):
        super(Relational, self).__init__(name=name, variable_key=variable_key)
        self.split = Split(axis=axis, size=2, reduction=relation_reduction)
        self.relation_unit = relation_unit
        self.reduction = Reduction(reduction=reduction)

    def forward(self, xs, y):
        xs >>= self.split
        xs = [(x, y) >> Concatenation() >> self.relation_unit for x in xs]
        return xs >> self.reduction


class Index(Unit):

    def __init__(self, name=None, variable_key=None):
        super(Index, self).__init__(name=name, variable_key=variable_key)

    def forward(self, x):
        index = None
        indexed_shape = shape(x)[1:-1]
        for n, dims in enumerate(indexed_shape):
            delta = 2.0 / (dims - 1)
            next_index = tf.range(start=-1.0, limit=(1.0 + 0.5 * delta), delta=delta, dtype=Model.dtype('float'))
            next_index = tf.expand_dims(input=next_index, axis=1)
            if index is None:
                index = next_index
            else:
                index = tf.stack(values=[index for _ in range(dims)], axis=n)
                for k, prev_dims in enumerate(indexed_shape[:n]):
                    next_index = tf.stack(values=[next_index for _ in range(prev_dims)], axis=k)
                index = tf.concat(values=(index, next_index), axis=(n + 1))
        index = tf.expand_dims(input=index, axis=0)
        multiples = [tf.shape(input=x)[0]] + [1] * (rank(x) - 1)
        index = tf.tile(input=index, multiples=multiples)
        index >>= Print(size=64)
        return tf.concat(values=(x, index), axis=(rank(x) - 1))


class Dense(Layer):

    def __init__(self, size, bias=False, normalization=True, nonlinearity='relu', dropout=False, name=None, variable_key=None):
        super(Dense, self).__init__(size=size, name=name, variable_key=variable_key)
        assert isinstance(bias, bool)
        assert isinstance(normalization, bool)
        assert not nonlinearity or Nonlinearity.valid(nonlinearity)
        assert not dropout or Dropout.valid(dropout)
        self.weights_init = nonlinearity or 'in-out'
        self.bias = Variable(key=self.subkey('bias'), shape=self.size, init='zeros') if bias else None
        self.normalization = Normalization(variable_key=self.subkey('normalization')) if normalization else Identity()
        self.nonlinearity = Nonlinearity(nonlinearity=nonlinearity) if nonlinearity else Identity()
        self.dropout = Dropout() if dropout else Identity()

    def forward(self, x):
        assert 2 <= rank(x) <= 4
        x >>= self.normalization
        x >>= self.nonlinearity
        if rank(x) == 2:
            self.weights = Variable(key=self.subkey('weights'), shape=(shape(x)[-1], self.size), init=self.weights_init)
            x = tf.matmul(a=x, b=self.weights())
        elif rank(x) == 3:
            self.weights = Variable(key=self.subkey('weights'), shape=(1, shape(x)[-1], self.size), init=self.weights_init)
            x = tf.nn.conv1d(value=x, filters=self.weights(), stride=1, padding='SAME')
        elif rank(x) == 4:
            self.weights = Variable(key=self.subkey('weights'), shape=(1, 1, shape(x)[-1], self.size), init=self.weights_init)
            x = tf.nn.conv2d(input=x, filter=self.weights(), strides=(1, 1, 1, 1), padding='SAME')
        if self.bias:
            x = tf.nn.bias_add(value=x, bias=self.bias())
        return x >> self.dropout


class Lstm(Layer):

    def __init__(self, size, initial_state_variable=False, name=None, variable_key=None):
        super(Lstm, self).__init__(size=size, name=name, variable_key=variable_key)
        self.initial_state = Variable(key=self.subkey('init'), shape=(1, 2, self.size), dtype='float') if initial_state_variable else None

    def forward(self, x=None, state=None):
        lstm = tf.contrib.rnn.LSTMCell(num_units=self.size)  #, state_is_tuple=False)
        if x is None:
            if self.initial_state is None:
                return lstm, lstm.zero_state(batch_size=1, dtype=Model.dtype('float'))
            else:
                initial_state = lambda t: tf.contrib.rnn.LSTMStateTuple(*tf.unstack(value=tf.tile(input=self.initial_state(), multiples=(tf.shape(input=t)[0], 1, 1)), axis=1))
                return lstm, initial_state
                # return lstm, self.initial_state()  # tf.contrib.rnn.LSTMStateTuple(c=initial_state[0], h=initial_state[1])
        else:
            return lstm(inputs=x, state=state)


class Gru(Layer):

    def __init__(self, size, initial_state_variable=True, name=None, variable_key=None):
        super(Gru, self).__init__(size=size, name=name, variable_key=variable_key)
        self.initial_state = Variable(key=self.subkey('init'), shape=(1, self.size), dtype='float') if initial_state_variable else None

    def forward(self, x=None, state=None):
        gru = tf.contrib.rnn.GRUCell(num_units=self.size)
        if x is None:
            if self.initial_state is None:
                return gru, gru.zero_state(batch_size=1, dtype=Model.dtype('float'))
            else:
                return gru, self.initial_state()  # tf.contrib.rnn.LSTMStateTuple(c=initial_state[0], h=initial_state[1])
        else:
            return gru(inputs=x, state=state)


class Rnn(Layer):

    def __init__(self, size, unit=Lstm, initial_state_variable=True, name=None, variable_key=None):
        super(Rnn, self).__init__(size=size, name=name, variable_key=variable_key)
        assert issubclass(unit, Layer)
        assert isinstance(initial_state_variable, bool)
        self.unit = unit(size=self.size, initial_state_variable=initial_state_variable, variable_key=self.subkey('unit'))

    def forward(self, x, length=None):
        if length is not None and rank(length) == 2:
            # asserts
            length = tf.squeeze(input=length, axis=1)
        unit, initial_state = self.unit()
        # initial_state = tf.tile(input=initial_state, multiples=(tf.shape(input=x)[0], 1))
        initial_state = initial_state(x)
        x, state = tf.nn.dynamic_rnn(cell=unit, inputs=x, sequence_length=length, dtype=Model.dtype('float'))  #, initial_state=initial_state)
        return x, tf.stack(values=(state.c, state.h), axis=1)


class Convolution(Layer):

    def __init__(self, size, window=(3, 3), transposed=False, bias=False, normalization=True, nonlinearity='relu', stride=1, padding='SAME', name=None, variable_key=None):
        super(Convolution, self).__init__(size=size, name=name, variable_key=variable_key)
        window = (window,) if isinstance(window, int) else tuple(window)
        assert 1 <= len(window) <= 2 and all(isinstance(n, int) and n > 0 for n in window)
        assert isinstance(transposed, bool) and (not transposed or len(window) == 2)
        assert isinstance(bias, bool)
        assert isinstance(normalization, bool)
        assert not nonlinearity or Nonlinearity.valid(nonlinearity)
        assert padding in ('SAME', 'VALID')
        self.window = window
        self.filters_init = nonlinearity or 'in-out'
        self.transposed = transposed
        self.bias = Variable(key=self.subkey('bias'), shape=(self.size,), init='zeros') if bias else None
        self.bias = bias
        self.normalization = Normalization(variable_key=self.subkey('normalization')) if normalization else Identity()
        self.nonlinearity = Nonlinearity(nonlinearity=nonlinearity) if nonlinearity else Identity()
        self.padding = padding
        if isinstance(stride, int):
            assert stride > 0
            self.stride = stride if len(window) == 1 else (1, stride, stride, 1)
        else:
            assert len(stride) == 2 and stride[0] > 0 and stride[1] > 0
            self.stride = (1, stride[0], stride[1], 1)

    def forward(self, x):
        x >>= self.normalization
        x >>= self.nonlinearity
        if self.transposed:
            filters_shape = self.window + (self.size, shape(x)[-1])
        else:
            filters_shape = self.window + (shape(x)[-1], self.size)
        self.filters = Variable(key=self.subkey('filters'), shape=filters_shape, init=self.filters_init)
        if len(self.window) == 1:
            x = tf.nn.conv1d(value=x, filters=self.filters(), stride=self.stride, padding=self.padding)
        elif self.transposed:
            batch, height, width, _ = shape(x)
            x = tf.nn.conv2d_transpose(value=x, filter=self.filters(), output_shape=(batch, height * self.stride[1], width * self.stride[2], self.size), strides=self.stride, padding=self.padding)
        else:
            x = tf.nn.conv2d(input=x, filter=self.filters(), strides=self.stride, padding=self.padding)
        if self.bias:
            x = tf.nn.bias_add(value=x, bias=self.bias())
        return x


class NgramConvolution(Layer):

    def __init__(self, size, ngrams=3, padding='VALID', name=None, variable_key=None):
        super(NgramConvolution, self).__init__(size=size, name=name, variable_key=variable_key)
        self.convolutions = []
        for ngram in range(1, ngrams + 1):  # not start with 1
            convolution = Convolution(size=self.size, window=ngram, normalization=False, nonlinearity='relu', padding=padding, variable_key=self.subkey(str(ngram) + 'gram'))  # norm, nonlin?
            self.convolutions.append(convolution)

    def forward(self, x):
        embeddings = [x >> convolution for convolution in self.convolutions]
        # requires SAME
        # phrase_embeddings = tf.stack(values=embeddings, axis=1)
        # phrase_embeddings = tf.reduce_max(input_tensor=phrase_embeddings, axis=???)
        # two reductions, and both concat only possible with SAME
        # maybe lstm?
        return tf.concat(values=embeddings, axis=1)


class Residual(Layer):

    def __init__(self, size, unit=Convolution, depth=2, reduction='sum', name=None, variable_key=None):
        super(Residual, self).__init__(size=size, name=name, variable_key=variable_key)
        assert issubclass(unit, Layer)
        assert isinstance(depth, int) and depth >= 0
        self.units = []
        for n in range(depth):
            self.units.append(unit(size=self.size, variable_key=self.subkey('unit' + str(n))))

    def forward(self, x):
        res = x
        for unit in self.units:
            res >>= unit
        return tf.add(x=x, y=res)


class Expand(Layer):

    def __init__(self, size, bottleneck=False, unit=Convolution, name=None, variable_key=None):
        super(Residual, self).__init__(size=size, name=name, variable_key=variable_key)
        assert issubclass(unit, Layer)
        self.bottleneck = unit(size=(self.size * bottleneck), variable_key=self.subkey('bottleneck')) if bottleneck else Identity()
        self.unit = unit(size=self.size, variable_key=self.subkey('unit'))

    def forward(self, x):
        fx = x >> self.bottleneck >> self.unit
        return tf.concat(values=(x, fx), axis=(rank(x) - 1))


class Fractal(Layer):

    def __init__(self, size, unit=Convolution, depth=3, reduction='mean', name=None, variable_key=None):
        super(Fractal, self).__init__(size=size, name=name, variable_key=variable_key)
        assert issubclass(unit, Layer)
        assert isinstance(depth, int) and depth >= 0
        self.depth = depth
        if depth > 0:
            self.fx = Fractal(size=self.size, unit=unit, depth=(depth - 1), variable_key=self.subkey('fractal' + str(depth)))
            self.ffx = Fractal(size=self.size, unit=unit, depth=(depth - 1), variable_key=self.subkey('fffractal' + str(depth)))
            self.unit = unit(size=self.size, variable_key=self.subkey('unit' + str(depth)))
            self.reduction = Reduction(reduction=reduction)

    def forward(self, x):
        if self.depth == 0:
            return x
        y = x >> self.fx >> self.ffx
        x >>= self.unit
        return (x, y) >> self.reduction


class Repeat(LayerStack):

    def __init__(self, layer, size, name=None, variable_key=None, **kwargs):
        super(Repeat, self).__init__(name=name, variable_key=variable_key)
        kwargs_list = [dict(size=s, variable_key=self.subkey('layer-{}'.format(n))) for n, s in enumerate(size)]
        for name, value in kwargs.items():
            if isinstance(value, list):
                assert len(value) == len(kwargs_list)
                for n in range(len(value)):
                    kwargs_list[n][name] = value[n]
            else:
                for n in range(len(kwargs_list)):
                    kwargs_list[n][name] = value
        for kwargs in kwargs_list:
            self.layers.append(layer(**dict(kwargs)))


class PoolingNet(LayerStack):

    def __init__(self, layer, sizes, depths, transition=None, pool='max', name=None, variable_key=None):
        super(PoolingNet, self).__init__(name=name, variable_key=variable_key)
        assert Pooling.valid(pool)
        n = 0
        for size, depth in zip(sizes, depths):
            if transition:
                self.layers.append(transition(size=size, variable_key=self.subkey('transition-{}'.format(depth))))
            for n in range(depth):
                self.layers.append(layer(size=size, variable_key=self.subkey('layer-{}-{}'.format(depth, n))))
                n += 1
            self.layers.append(Pooling(pool=pool))


class ConvolutionalNet(PoolingNet):

    def __init__(self, sizes, depths, name=None, variable_key=None):
        super(ConvolutionalNet, self).__init__(layer=Convolution, sizes=sizes, depths=depths, name=name, variable_key=variable_key)


class ResidualNet(PoolingNet):

    def __init__(self, sizes, depths, name=None, variable_key=None):  # Residual configurable which layer etc
        super(ResidualNet, self).__init__(layer=Residual, sizes=sizes, depths=depths, transition=Convolution, name=name, variable_key=variable_key)


class DenseNet(PoolingNet):

    def __init__(self, sizes, depths, name=None, variable_key=None):  # Residual configurable which layer etc
        super(DenseNet, self).__init__(layer=Expand, sizes=sizes, depths=depths, transition=Convolution, name=name, variable_key=variable_key)


class FractalNet(PoolingNet):

    def __init__(self, sizes, name=None, variable_key=None):  # Fractal configurable which layer etc
        super(FractalNet, self).__init__(layer=Fractal, sizes=sizes, depths=([1] * len(sizes)), transition=Convolution, name=name, variable_key=variable_key)


FullyConnected = Full = FC = Dense

Linear = (lambda size, bias=False, name=None, variable_key=None: Dense(size=size, bias=bias, normalization=False, nonlinearity=None, dropout=False, name=name, variable_key=variable_key))
