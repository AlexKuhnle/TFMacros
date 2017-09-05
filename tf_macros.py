from itertools import chain, combinations
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


def make_least_common_shape(xs, ignore_ranks=()):
    assert len(xs) > 1
    shapes = [shape(x) for x in xs]
    common_rank = len(shapes[0])
    assert all(len(s) == common_rank for s in shapes)
    ref_shape = tuple(max(s[r] for s in shapes) for r in range(common_rank))
    ys = list()
    for x, s in zip(xs, shapes):
        multiples = [ref_dims if dims == 1 and r not in ignore_ranks else 1 for r, (dims, ref_dims) in enumerate(zip(s, ref_shape))]
        if not all(m == 1 for m in multiples):
            x = tf.tile(input=x, multiples=multiples)
        assert rank(x) == common_rank and all(d1 == d2 for r, (d1, d2) in enumerate(zip(shape(x), ref_shape)) if r not in ignore_ranks)
        ys.append(x)
    return ys


def make_broadcastable(xs):
    assert len(xs) > 0
    if len(xs) == 1 and not isinstance(xs[0], tf.Tensor):
        xs = xs[0]
    shapes = [shape(x) for x in xs]
    ref_shape = max(shapes, key=(lambda s: len(s) - sum(dims == 1 for dims in s) / (len(s) + 1)))
    ys = list()
    for x, s in zip(xs, shapes):
        s = list(s)
        if len(s) < len(ref_shape):
            last_dims = None
            for r, (dims, ref_dims) in enumerate(zip(s, ref_shape)):
                if r < len(s) and dims in (ref_dims, 1):
                    last_dims = dims
                else:
                    assert dims != last_dims
                    x = tf.expand_dims(input=x, axis=r)
                    s.insert(r, 1)
        assert rank(x) == len(ref_shape) and all(d1 == d2 or d1 == 1 for d1, d2 in zip(shape(x), ref_shape))
        ys.append(x)
    return ys


class Model(object):

    precision = 32
    current = None

    @staticmethod
    def dtype(dtype, include_bytes=False):
        assert Model.precision % 8 == 0
        assert dtype in ('float', 'int')
        if dtype == 'float':
            if Model.precision == 32:
                dtype = tf.float32
            else:
                assert False
        elif dtype == 'int':
            if Model.precision == 32:
                dtype = tf.int32
            else:
                assert False
        else:
            assert False
        if include_bytes:
            return dtype, Model.precision // 8
        else:
            return dtype

    def __init__(self, name=None, optimizer='adam', learning_rate=0.001, weight_decay=0.0, clip_gradients=1.0, model_file=None, summary_file=None):
        assert name is None or isinstance(name, str)
        assert optimizer in ('adam',)
        assert isinstance(learning_rate, float)
        assert isinstance(weight_decay, float)
        assert isinstance(clip_gradients, float)
        assert model_file is None or isinstance(model_file, str)
        assert summary_file is None or isinstance(summary_file, str)
        self.name = name
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_gradients = clip_gradients
        self.model_file = model_file
        self.summary_file = summary_file
        self.tensors = dict()
        self.variables = dict()
        self.placeholders = dict()
        self.num_parameters = 0
        self.num_bytes = 0
        self.session = None
        self.defined = False

    def __str__(self):
        if self.name is None:
            return 'Model'
        else:
            return self.name

    def register_tensor(self, key, tensor):
        assert key not in ('loss', 'dropout')
        assert key not in self.tensors
        self.tensors[key] = tensor

    def register_variable(self, key, variable, num_parameters, num_bytes):
        if key in self.variables:
            assert variable == self.variables[key]
        else:
            self.variables[key] = variable
            self.num_parameters += num_parameters
            self.num_bytes += num_bytes

    def register_placeholder(self, key, placeholder):
        assert key not in self.placeholders
        self.placeholders[key] = placeholder

    def __enter__(self):
        assert Model.current is None
        Model.current = self
        self.scope = tf.variable_scope(str(self))
        self.scope.__enter__()
        Input(name='dropout', shape=(), batched=False).forward()
        self.dropout = self.placeholders.pop('dropout')
        return self

    def __exit__(self, type, value, tb):
        if type is not None:
            raise
        if self.defined:
            self.coordinator.request_stop()
            self.coordinator.join(threads=self.queue_threads)
            if self.model_file:
                self.saver.save(sess=self.session, save_path=self.model_file)
        else:
            for name, variable in self.variables.items():
                regularization = self.weight_decay * tf.nn.l2_loss(t=variable, name=(name + '-regularization'))
                tf.losses.add_loss(loss=regularization, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.losses.get_total_loss()
            self.tensors['loss'] = loss
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            grads_and_vars = optimizer.compute_gradients(loss=loss)
            grads_and_vars = [(tf.clip_by_value(t=grad, clip_value_min=-self.clip_gradients, clip_value_max=self.clip_gradients), var) for grad, var in grads_and_vars]
            self.optimization = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
            self.scope.__exit__(type, value, tb)
            assert False
        assert Model.current is not None
        Model.current = None

    def finalize(self, restore=False):
        for name, variable in self.variables.items():
            regularization = self.weight_decay * tf.nn.l2_loss(t=variable, name=(name + '-regularization'))
            tf.losses.add_loss(loss=regularization, loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.losses.get_total_loss()
        self.tensors['loss'] = loss
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss=loss)
        grads_and_vars = [(tf.clip_by_value(t=grad, clip_value_min=-self.clip_gradients, clip_value_max=self.clip_gradients), var) for grad, var in grads_and_vars]
        self.optimization = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
        global_variables_initializer = tf.global_variables_initializer()
        if self.model_file is not None:
            self.saver = tf.train.Saver()
        if self.summary_file is not None:
            tf.summary.scalar(name='loss', tensor=loss)
            for variable in tf.trainable_variables():
                tf.summary.histogram(name=variable.name, values=variable)
            self.summaries = tf.summary.merge_all()
        self.scope.__exit__(None, None, None)
        tf.get_default_graph().finalize()
        self.defined = True

        self.session = tf.Session()
        if restore:
            assert self.model_file
            self.saver.restore(sess=self.session, save_path=self.model_file)
        else:
            self.session.run(fetches=global_variables_initializer)
        if self.summary_file is not None:
            self.summary_writer = tf.summary.FileWriter(logdir=self.summary_file, graph=self.session.graph)
        self.coordinator = tf.train.Coordinator()
        self.queue_threads = tf.train.start_queue_runners(sess=self.session, coord=self.coordinator)

    def __call__(self, query=None, data=None, optimize=True, summarize=True, dropout=0.0):
        assert self.session
        if query is None:
            fetches = dict()
        elif isinstance(query, str):
            fetches = dict(query=self.tensors[query])
        else:
            fetches = {name: self.tensors[name] for name in query}
        if optimize:
            assert 'optimization' not in fetches
            fetches['optimization'] = self.optimization
        if self.summary_file is not None and summarize:
            assert 'summaries' not in fetches
            fetches['summaries'] = self.summaries
        if data is None:
            feed_dict = None
        elif isinstance(data, dict):
            feed_dict = {self.placeholders[name]: value for name, value in data.items() if name in self.placeholders}
        else:
            assert len(self.placeholders) == 1
            feed_dict = {next(iter(self.placeholders.values())): data}
        if dropout > 0.0:
            assert dropout < 1.0
            feed_dict[self.dropout] = dropout
        fetched = self.session.run(fetches=fetches, feed_dict=feed_dict)
        if optimize:
            fetched.pop('optimization')
        if self.summary_file is not None and summarize:
            fetched.pop('summaries')
        return fetched


class Unit(object):

    num_in = -1
    num_out = -1

    _index = 0

    def __init__(self, name=None):
        assert Model.current is not None
        assert name is None or isinstance(name, str)
        if name is None:
            name = self.__class__.__name__ + str(self.__class__._index)
            self.__class__._index += 1
        self.name = name
        self.outputs = dict()
        self._forward = tf.make_template(name_=str(self), func_=self.forward)

    def forward(self, *inputs):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def __call__(self, inputs=(), output_key=None):
        assert output_key is None or isinstance(output_key, str)
        if output_key is not None and output_key in self.outputs:
            return self.outputs[output_key]
        output = self._forward(*inputs)
        if isinstance(output, tf.Tensor):
            if output_key is not None:
                self.outputs[output_key] = output
                Model.current.register_tensor(key=output_key, tensor=output)
        elif len(output) == 1:
            output = output[0]
            if output_key is not None:
                self.outputs[output_key] = output
                Model.current.register_tensor(key=output_key, tensor=output)
        else:
            output = tuple(output)
            if output_key is not None:
                self.outputs[output_key] = output
                for n, tensor in enumerate(output):
                    Model.current.register_tensor(key=(output_key + str(n)), tensor=tensor)
        return output

    def __rshift__(self, other):
        assert isinstance(other, Unit)
        if self.__class__.num_in == 0:
            inputs = self()
            inputs = (inputs,) if isinstance(inputs, tf.Tensor) else inputs
            return other(inputs=inputs)
        else:
            return Composed(first=self, second=other)

    def __rrshift__(self, other):
        composed = False
        if isinstance(other, tf.Tensor):
            return self(inputs=(other,))
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
            return self(inputs=inputs)


class Composed(Unit):

    def __init__(self, first, second):
        self.first = first
        self.second = second
        super(Composed, self).__init__()

    def forward(self, *xs):
        assert isinstance(self.first, Unit) or len(xs) == 1
        if len(xs) == 1:
            return xs[0] >> self.first >> self.second
        else:
            return xs >> self.first >> self.second

    def __rshift__(self, other):
        assert isinstance(other, Unit)
        return Composed(first=self, second=other)


class Layer(Unit):

    num_in = 1
    num_out = 1

    def __init__(self, size, name=None):
        assert self.__class__.num_in == 1 and self.__class__.num_out == 1
        assert isinstance(size, int) and size > 0
        self.size = size
        super(Layer, self).__init__(name=name)


class LayerStack(Unit):

    def __init__(self, name=None):
        assert hasattr(self, 'layers')
        super(LayerStack, self).__init__(name=name)

    def forward(self, *xs):
        for layer in self.layers:
            xs >>= layer
        return xs


class Variable(Unit):

    num_in = 0
    num_out = 1

    def __init__(self, name, shape=None, dtype='float', init='out', value=None):
        assert self.__class__.num_in == 0 and self.__class__.num_out == 1
        assert isinstance(name, str)
        if shape is not None:
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            assert len(shape) > 0 and all(isinstance(n, int) and n > 0 for n in shape)
        assert init in ('constant', 'zeros', 'ones', 'in', 'out', 'in-out', 'stddev') or Activation.valid(init)
        assert init in ('constant', 'zeros', 'ones') or dtype == 'float'
        self.shape = shape
        self.dtype, self.dtype_bytes = Model.dtype(dtype=dtype, include_bytes=True)
        self.init = init
        self.value = value
        super(Variable, self).__init__(name=name)

    def specify_shape(self, shape):
        if self.shape is None:
            self.shape = shape
        else:
            assert self.shape == shape

    def forward(self):
        assert self.shape is not None
        if self.init == 'zeros':
            initializer = tf.zeros_initializer(dtype=self.dtype)
        elif self.init == 'ones':
            initializer = tf.ones_initializer(dtype=self.dtype)
        elif self.init == 'stddev':
            assert self.value is not None
            initializer = tf.random_normal_initializer(mean=0.0, stddev=self.value, dtype=tf.float32)
        elif self.init == 'selu':
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_OUT', dtype=self.dtype)
        elif self.init == 'out':
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', dtype=self.dtype)
        elif self.init == 'in' or self.init in ('elu', 'relu'):
            assert len(self.shape) >= 2
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', dtype=self.dtype)
        elif self.init == 'in-out' or Activation.valid(self.init):
            assert len(self.shape) >= 2
            initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', dtype=self.dtype)
        else:
            assert False
        variable = tf.get_variable(name=self.name, shape=self.shape, dtype=self.dtype, initializer=initializer)
        num_parameters = product(self.shape)
        num_bytes = num_parameters * self.dtype_bytes
        Model.current.register_variable(key='{}/{}'.format(tf.get_variable_scope().name, self.name), variable=variable, num_parameters=num_parameters, num_bytes=num_bytes)
        return tf.identity(input=variable)


class Input(Unit):

    num_in = 0
    num_out = 1

    def __init__(self, name, shape, dtype='float', batched=True, tensor=None):
        assert isinstance(name, str)
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        assert all(isinstance(n, int) and (n > 0 or n == -1) for n in shape)
        assert isinstance(batched, bool)
        self.shape = tuple(None if x == -1 else x for x in shape)
        self.dtype = Model.dtype(dtype=dtype)
        if batched:
            self.shape = (None,) + self.shape
        self.tensor = tensor
        super(Input, self).__init__(name=name)

    def forward(self):
        if self.tensor is None:
            placeholder = tf.placeholder(dtype=self.dtype, shape=self.shape, name=self.name)
            Model.current.register_placeholder(key=self.name, placeholder=placeholder)
            self.tensor = tf.identity(input=placeholder)
        return self.tensor


class Binary(Input):

    num_in = 1

    def __init__(self, name, soft=0.0, tensor=None):
        assert isinstance(name, str)
        assert isinstance(soft, float) and 0.0 <= soft < 0.5
        self.soft = soft
        super(Binary, self).__init__(name=name, shape=(), batched=True, tensor=tensor)

    def forward(self, x):
        correct = y = super(Binary, self).forward()
        if self.soft > 0.0:
            noise = tf.random_uniform(shape=(1, 1), minval=0.0, maxval=self.soft)
            correct = tf.abs(x=(correct - noise))
        x >>= Linear(size=0)
        x = (tf.tanh(x=x) + 1.0) / 2.0
        cross_entropy = -(correct * tf.log(x=x + 1e-10) + (1.0 - correct) * tf.log(x=1.0 - x + 1e-10))
        loss = tf.reduce_mean(input_tensor=cross_entropy)
        tf.losses.add_loss(loss=loss)
        prediction = tf.cast(x=tf.greater(x=x, y=tf.constant(value=0.5)), dtype=Model.dtype('float'))
        correct = tf.cast(x=tf.equal(x=prediction, y=correct), dtype=Model.dtype('float'))
        accuracy = tf.reduce_mean(input_tensor=correct)
        Model.current.register_tensor(key=(self.name + '-accuracy'), tensor=accuracy)
        return y


class Classification(Input):

    def __init__(self, name, num_classes, multi_class=False, soft=0.0, tensor=None):
        assert isinstance(num_classes, int) and num_classes > 0
        assert isinstance(multi_class, bool)
        assert isinstance(soft, float) and 0.0 <= soft < 0.5
        self.num_classes = num_classes
        self.multi_class = multi_class
        self.soft = soft
        super(Classification, self).__init__(name=name, shape=(num_classes,), batched=True, tensor=tensor)

    def forward(self, x):
        correct = y = super(Classification, self).forward()
        if not self.multi_class and rank(correct) == 1:
            correct = tf.one_hot(indices=correct, depth=self.num_classes)
        if self.soft > 0.0:
            noise = tf.random_uniform(shape=(1, shape(correct)[1]), minval=0.0, maxval=self.soft)
            correct = tf.abs(x=(correct - noise))
        x >>= Linear(size=self.num_classes)
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
        Model.current.register_tensor(key=(self.name + '-precision'), tensor=precision)
        Model.current.register_tensor(key=(self.name + '-recall'), tensor=recall)
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

    def __init__(self, size=10, times=None, prefix=None, name=None):
        assert isinstance(size, int) and size > 0
        assert times is None or isinstance(times, int) and times > 0
        assert prefix is None or isinstance(prefix, str)
        self.size = size
        self.times = times
        self.prefix = prefix
        super(Print, self).__init__(name=name)

    def forward(self, *xs):
        if self.prefix is None or self.prefix[-2:] == ': ':
            message = self.prefix
        elif self.prefix[-1] == ':':
            message = self.prefix + ' '
        else:
            message = self.prefix + ': '
        return (tf.Print(input_=xs[0], data=xs, message=message, first_n=self.times, summarize=self.size),) + tuple(xs[1:])


class Select(Unit):

    def __init__(self, index, name=None):
        assert isinstance(index, int) and index >= 0
        self.index = index
        super(Select, self).__init__(name=name)

    def forward(self, *xs):
        assert len(xs) > self.index
        return xs[self.index]


class Activation(Unit):

    @staticmethod
    def valid(activation):
        return activation in ('elu', 'relu', 'sigmoid', 'softmax', 'tanh')

    def __init__(self, activation='relu', name=None):
        assert Activation.valid(activation)
        self.activation = activation
        super(Activation, self).__init__(name=name)

    def forward(self, x):
        if self.activation == 'elu':
            return tf.nn.elu(features=x)
        elif self.activation == 'relu':
            return tf.nn.relu(features=x)
        elif self.activation == 'selu':
            # https://arxiv.org/pdf/1706.02515.pdf
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.where(condition=(x >= 0.0), x=x, y=(alpha * tf.nn.elu(features=x)))
        elif self.activation == 'sigmoid':
            return tf.sigmoid(x=x)
        elif self.activation == 'softmax':
            return tf.nn.softmax(logits=x)
        elif self.activation == 'tanh':
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

    def __init__(self, offset=False, scale=False, variance_epsilon=1e-6, name=None):
        assert isinstance(offset, bool) or issubclass(offset, Layer)
        assert isinstance(scale, bool) or issubclass(scale, Layer)
        assert isinstance(variance_epsilon, float) and variance_epsilon > 0.0
        self.offset = offset
        self.scale = scale
        self.variance_epsilon = variance_epsilon
        super(Normalization, self).__init__(name=name)

    def forward(self, x, condition=None):
        mean, variance = tf.nn.moments(x=x, axes=tuple(range(rank(x) - 1)))
        if condition is None:
            if self.offset:
                self.offset = Variable(name='offset', shape=shape(mean), init='zeros')
                offset = self.offset()
            else:
                self.offset = offset = None
            if self.scale:
                self.scale = Variable(name='scale', shape=shape(mean), init='zeros')
                scale = 1.0 + self.scale()
            else:
                self.scale = scale = None
        else:
            assert issubclass(self.offset, Layer) and issubclass(self.scale, Layer)
            size = shape(mean)[0]
            self.offset = self.offset(size=size)
            offset = condition >> self.offset
            offset = tf.expand_dims(input=tf.expand_dims(input=offset, axis=1), axis=1)
            self.scale = self.scale(size=size)
            scale = 1.0 + (condition >> self.scale)
            scale = tf.expand_dims(input=tf.expand_dims(input=scale, axis=1), axis=1)
        return tf.nn.batch_normalization(x=x, mean=mean, variance=variance, offset=offset, scale=scale, variance_epsilon=self.variance_epsilon)


class Reduction(Unit):

    num_in = -1
    num_out = 1

    @staticmethod
    def valid(reduction):
        return isinstance(reduction, str) and reduction in ('cbp', 'collapse', 'concat', 'conv', 'conv2d', 'last', 'max', 'mean', 'min', 'prod', 'stack', 'sum')

    def __init__(self, reduction, axis=-1, arg=-1, name=None):
        assert Reduction.valid(reduction)
        if isinstance(axis, int):
            axis = (axis,)
        elif len(axis) == 3 and axis[1] is Ellipsis:
            assert isinstance(axis[0], int) and isinstance(axis[2], int)
            axis = tuple(axis)
        else:
            assert len(axis) > 0 and all(isinstance(a, int) for a in axis)
            axis = tuple(sorted(axis))
        assert len(set(axis)) == len(axis)
        assert isinstance(arg, int)
        self.reduction = reduction
        self.axis = axis
        self.arg = arg
        self.multiple_inputs = None
        if reduction in ('conv', 'conv2d'):
            self.weights = Variable(name='weights', init='in-out')
        super(Reduction, self).__init__(name=name)

    def forward(self, *xs):
        assert len(xs) > 0
        if self.multiple_inputs is None:
            self.multiple_inputs = len(xs) > 1

        if self.multiple_inputs:
            assert self.axis == (-1,)
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

            elif self.reduction in ('collapse', 'conv', 'conv2d'):
                xs = make_least_common_shape(xs=xs)
                x = tf.stack(values=xs, axis=axis)

        else:
            x = xs[0]

            if len(self.axis) == 3 and self.axis[1] is Ellipsis:
                start, _, end = self.axis
                start = start if start >= 0 else rank(x) + start
                end = end if end >= 0 else rank(x) + end
                self.axis = tuple(range(start, end + 1))
            elif any(a < 0 for a in self.axis):
                self.axis = tuple(sorted(a if a >= 0 else rank(x) + a for a in self.axis))
                assert len(set(self.axis)) == len(self.axis)
            assert self.axis[0] >= 0 and self.axis[-1] < rank(x)

            if self.reduction in ('concat', 'stack'):
                for axis in reversed(self.axis):
                    xs = [y for x in xs for y in tf.unstack(value=x, axis=axis)]

        if self.reduction in ('concat', 'stack'):
            self.arg = self.arg if self.arg >= 0 else rank(xs[0]) + self.arg
            assert 0 <= self.arg < rank(xs[0])
            xs = make_least_common_shape(xs=xs, ignore_ranks=(self.arg,))

        if self.reduction == 'collapse':
            assert all(axis == n for n, axis in enumerate(self.axis, self.axis[0]))
            reduced_shape = shape(x)
            reduced_shape = reduced_shape[:self.axis[0]] + (product(reduced_shape[self.axis[0]:self.axis[-1]]),) + reduced_shape[self.axis[-1]:]
            return tf.reshape(tensor=x, shape=reduced_shape)

        elif self.reduction == 'concat':
            return tf.concat(values=xs, axis=self.arg)

        elif self.reduction == 'conv':
            assert self.axis == (-1,) or (len(self.axis) == rank(x) - 2 and all(axis == n for n, axis in enumerate(self.axis, 1)))
            reduced_shape = shape(x)
            reduced_shape = (reduced_shape[:1], product(reduced_shape[1:-1]), reduced_shape[-1:])
            x = tf.transpose(a=tf.reshape(tensor=x, shape=reduced_shape), perm=(0, 2, 1))
            self.weights.specify_shape(shape=(1, shape(x)[-1], 1))
            x = tf.nn.conv1d(value=x, filters=self.weights(), stride=1, padding='SAME')
            return tf.squeeze(input=x, axis=2)

        elif self.reduction == 'conv2d':
            assert self.axis == (-1,) or self.axis == (1, 2)
            self.weights.specify_shape(shape=(shape(x)[1], shape(x)[2], shape(x)[-1], shape(x)[-1]))
            x = tf.nn.conv2d(input=x, filter=self.weights(), strides=(1, 1, 1, 1), padding='VALID')
            return tf.squeeze(input=tf.squeeze(input=x, axis=2), axis=1)

        elif self.reduction == 'last':
            if self.multiple_inputs:
                return xs[-1]
            else:
                for axis in reversed(self.axis):
                    if axis == 0:
                        x = x[-1, Ellipsis]
                    elif axis == 1:
                        x = x[:, -1, Ellipsis]
                    elif axis == 2:
                        x = x[:, :, -1, Ellipsis]
                    elif axis == 3:
                        x = x[:, :, :, -1, Ellipsis]
                    elif axis == 4:
                        x = x[:, :, :, :, -1, Ellipsis]
                    else:
                        assert False
                return x

        elif self.reduction == 'max':
            return tf.reduce_max(input_tensor=x, axis=self.axis)

        elif self.reduction == 'mean':
            return tf.reduce_mean(input_tensor=x, axis=self.axis)

        elif self.reduction == 'min':
            return tf.reduce_min(input_tensor=x, axis=self.axis)

        elif self.reduction == 'prod':
            return tf.reduce_prod(input_tensor=x, axis=self.axis)

        elif self.reduction == 'stack':
            return tf.stack(values=xs, axis=self.arg)

        elif self.reduction == 'sum':
            return tf.reduce_sum(input_tensor=x, axis=self.axis)


# class Concatenation(Unit):

#     def __init__(self, axis=-1, name=None):
#         assert isinstance(axis, int)
#         self.axis = axis
#         super(Concatenation, self).__init__(name=name)

#     def forward(self, *xs):
#         assert len(xs) >= 1 and all(rank(x) == rank(xs[0]) for x in xs)  # what if length 0?
#         axis = self.axis if self.axis >= 0 else rank(xs[0]) + self.axis
#         assert rank(xs[0]) > axis
#         return tf.concat(values=xs, axis=axis)


# class Stack(Unit):

#     def __init__(self, axis=-1, name=None):
#         assert isinstance(axis, int)
#         self.axis = axis
#         super(Stack, self).__init__(name=name)

#     def forward(self, *xs):
#         assert len(xs) >= 1 and all(rank(x) == rank(xs[0]) for x in xs)
#         axis = self.axis if self.axis >= 0 else rank(xs[0]) + self.axis + 1
#         assert rank(xs[0]) >= axis
#         return tf.stack(values=xs, axis=axis)


class Attention(Unit):

    def __init__(self, assessment, name=None):
        assert isinstance(assessment, Unit)
        self.assessment = assessment
        self.softmax = Activation(activation='softmax')
        self.reduction = Reduction(reduction='sum', axis=(1, Ellipsis, -2))
        super(Attention, self).__init__(name=name)

    def forward(self, x, query):
        assert rank(x) > 2 and rank(query) == 2 and shape(query)[0] == shape(x)[0]
        for _ in range(rank(x) - 2):
            query = tf.expand_dims(input=query, axis=1)
        attention = (x, query) >> self.assessment >> self.softmax
        assert shape(attention) == shape(x)[:-1]
        attention = tf.expand_dims(input=attention, axis=(rank(x) - 1))
        return (x * attention) >> self.reduction


class CompactBilinearPooling(Unit):

    def __init__(self, size=None, name=None):
        # assert GPU !!!
        # default arg size
        assert size is None or (isinstance(size, int) and size > 0)
        self.size = size
        super(CompactBilinearPooling, self).__init__(name=name)

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
            self.sketch_indices = Variable(name=('indices' + str(n)), shape=input_size, init='constant', value=sketch_indices)
            sketch_indices = tf.expand_dims(input=self.sketch_indices(), axis=1)
            sketch_indices = tf.concat(values=(indices, sketch_indices), axis=1)
            sketch_values = tf.random_uniform(shape=(input_size,))
            self.sketch_values = Variable(name=('values' + str(n)), shape=input_size, init='constant', value=sketch_values)
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
                x, p = make_broadcastable(xs=(x, p))
                p = tf.multiply(x=p, y=x)
        return tf.ifft(input=tf.real(input=p))


class Pooling(Unit):

    @staticmethod
    def valid(pool):
        return isinstance(pool, str) and pool in ('none', 'average', 'avg', 'max', 'maximum')

    def __init__(self, pool='max', window=(2, 2), stride=2, padding='SAME', name=None):
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
        super(Pooling, self).__init__(name=name)

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

    def __init__(self, indices, size, name=None):
        assert isinstance(indices, int) and indices > 0
        assert isinstance(size, int) and size > 0
        self.embeddings = Variable(name='embeddings', shape=(indices, size))
        super(Embedding, self).__init__(name=name)

    def forward(self, x):
        return tf.nn.embedding_lookup(params=self.embeddings(), ids=x)


class Split(Unit):

    def __init__(self, axis=1, size=1, reduction=None, name=None):
        axis = (axis,) if isinstance(axis, int) else tuple(sorted(axis, reverse=True))
        size = (size,) if isinstance(size, int) else tuple(size)
        assert all(isinstance(a, int) and a >= 0 for a in axis)
        assert all(isinstance(s, int) and s > 0 for s in size)
        self.axis = axis
        self.size = size
        self.reduction = None if reduction is None else Reduction(reduction=reduction)
        super(Split, self).__init__(name=name)

    def forward(self, x):
        xs = [x]
        for a in self.axis:
            xs = [y for x in xs for y in tf.unstack(value=x, axis=a)]
        if self.size != (1,):
            xs = chain(*(combinations(xs, r=s) for s in self.size))  # others interesting? permutation, product?
        if self.reduction is not None:
            xs = [x >> self.reduction for x in xs]
        return tuple(xs)


class Relational(Unit):

    def __init__(self, relation_unit, axis=1, relation_reduction='concat', reduction='sum', name=None):
        self.split = Split(axis=axis, size=2, reduction=relation_reduction)
        self.relation_unit = relation_unit
        self.reduction = Reduction(reduction=reduction)
        super(Relational, self).__init__(name=name)

    def forward(self, xs, y):
        xs >>= self.split
        xs = [(x, y) >> Reduction(reduction='concat') >> self.relation_unit for x in xs]
        return xs >> self.reduction


class Index(Unit):

    def __init__(self, name=None):
        super(Index, self).__init__(name=name)

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
        return tf.concat(values=(x, index), axis=(rank(x) - 1))


class Linear(Layer):

    def __init__(self, size, bias=False, name=None):
        assert isinstance(bias, bool)
        if size == 0:
            size = 1
            self.squeeze = True
        else:
            self.squeeze = False
        self.bias = Variable(name='bias', shape=size, init='zeros') if bias else None
        super(Linear, self).__init__(size=size, name=name)

    def forward(self, x):
        assert 2 <= rank(x) <= 4
        if rank(x) == 2:
            self.weights = Variable(name='weights', shape=(shape(x)[-1], self.size), init='in-out')
            x = tf.matmul(a=x, b=self.weights())
        elif rank(x) == 3:
            self.weights = Variable(name='weights', shape=(1, shape(x)[-1], self.size), init='in-out')
            x = tf.nn.conv1d(value=x, filters=self.weights(), stride=1, padding='SAME')
        elif rank(x) == 4:
            self.weights = Variable(name='weights', shape=(1, 1, shape(x)[-1], self.size), init='in-out')
            x = tf.nn.conv2d(input=x, filter=self.weights(), strides=(1, 1, 1, 1), padding='SAME')
        if self.bias:
            x = tf.nn.bias_add(value=x, bias=self.bias())
        if self.squeeze:
            x = tf.squeeze(input=x, axis=-1)
        return x


class Dense(Layer):

    def __init__(self, size, bias=False, normalization=True, activation='tanh', gated=False, norm_act_before=False, dropout=False, name=None):
        assert isinstance(bias, bool)
        assert isinstance(normalization, bool)
        assert not activation or Activation.valid(activation)
        assert isinstance(gated, bool)
        assert isinstance(norm_act_before, bool)
        assert not dropout or Dropout.valid(dropout)
        if size == 0:
            size = 1
            self.squeeze = True
        else:
            self.squeeze = False
        self.weights = Variable(name='weights', init=(activation or 'in-out'))
        self.bias = Variable(name='bias', shape=size, init='zeros') if bias else None
        self.normalization = Normalization() if normalization else Identity()
        self.activation = Activation(activation=activation) if activation else Identity()
        self.gated = gated
        if gated:
            self.gate_weights = Variable(name='weights', init='sigmoid')
            self.gate_bias = Variable(name='bias', shape=size, init='zeros') if bias else None
            self.gate_activation = Activation(activation='sigmoid')
        self.norm_act_before = norm_act_before
        self.dropout = Dropout() if dropout else Identity()
        super(Dense, self).__init__(size=size, name=name)

    def forward(self, x):
        assert 2 <= rank(x) <= 4
        if self.norm_act_before:
            x >>= self.normalization
            x >>= self.activation
        if rank(x) == 2:
            self.weights.specify_shape(shape=(shape(x)[-1], self.size))
            x = tf.matmul(a=x, b=self.weights())
            if self.gated:
                self.gate_weights.specify_shape(shape=(shape(x)[-1], self.size))
                gate = tf.matmul(a=x, b=self.gate_weights())
        elif rank(x) == 3:
            self.weights.specify_shape(shape=(1, shape(x)[-1], self.size))
            x = tf.nn.conv1d(value=x, filters=self.weights(), stride=1, padding='SAME')
            if self.gated:
                self.gate_weights.specify_shape(shape=(1, shape(x)[-1], self.size))
                gate = tf.nn.conv1d(value=x, filters=self.gate_weights(), stride=1, padding='SAME')
        elif rank(x) == 4:
            self.weights.specify_shape(shape=(1, 1, shape(x)[-1], self.size))
            x = tf.nn.conv2d(input=x, filter=self.weights(), strides=(1, 1, 1, 1), padding='SAME')
            if self.gated:
                self.gate_weights.specify_shape(shape=(1, 1, shape(x)[-1], self.size))
                gate = tf.nn.conv2d(input=x, filter=self.gate_weights(), strides=(1, 1, 1, 1), padding='SAME')
        if self.bias is not None:
            x = tf.nn.bias_add(value=x, bias=self.bias())
            if self.gated:
                gate = tf.nn.bias_add(value=gate, bias=self.gate_bias())
        if self.squeeze:
            x = tf.squeeze(input=x, axis=-1)
            if self.gated:
                gate = tf.squeeze(input=gate, axis=-1)
        if not self.norm_act_before:
            x >>= self.normalization
            x >>= self.activation
        if self.gated:
            x *= (gate >> self.gate_activation)
        return x >> self.dropout


class Lstm(Layer):

    # variables not registered !!!

    def __init__(self, size, initial_state_variable=False, name=None):
        self.initial_state = Variable(name='init', shape=(1, 2, size), dtype='float') if initial_state_variable else None
        super(Lstm, self).__init__(size=size, name=name)

    def forward(self, x=None, state=None):
        lstm = tf.contrib.rnn.LSTMCell(num_units=self.size)  # state_is_tuple=False)
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

    # variables not registered !!!

    def __init__(self, size, initial_state_variable=True, name=None):
        self.initial_state = Variable(name='init', shape=(1, size), dtype='float') if initial_state_variable else None
        super(Gru, self).__init__(size=size, name=name)

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

    def __init__(self, size, unit=Lstm, initial_state_variable=True, name=None):
        assert issubclass(unit, Layer)
        assert isinstance(initial_state_variable, bool)
        self.unit = unit(size=size, initial_state_variable=initial_state_variable)
        super(Rnn, self).__init__(size=size, name=name)

    def forward(self, x, length=None):
        if length is not None and rank(length) == 2:
            # asserts
            length = tf.squeeze(input=length, axis=1)
        unit, initial_state = self.unit()
        # initial_state = tf.tile(input=initial_state, multiples=(tf.shape(input=x)[0], 1))
        initial_state = initial_state(x)
        x, state = tf.nn.dynamic_rnn(cell=unit, inputs=x, sequence_length=length, dtype=Model.dtype('float'))  # initial_state=initial_state)
        return x, tf.stack(values=(state.c, state.h), axis=1)


class Convolution(Layer):

    def __init__(self, size, window=(3, 3), transposed=False, bias=False, normalization=True, activation='relu', norm_act_before=False, stride=1, padding='SAME', name=None):
        window = (window,) if isinstance(window, int) else tuple(window)
        assert 1 <= len(window) <= 2 and all(isinstance(n, int) and n > 0 for n in window)
        assert isinstance(transposed, bool) and (not transposed or len(window) == 2)
        assert isinstance(bias, bool)
        assert isinstance(normalization, bool) or isinstance(normalization, tuple)
        assert not activation or Activation.valid(activation)
        assert isinstance(norm_act_before, bool)
        assert padding in ('SAME', 'VALID')
        if size == 0:
            size = 1
            self.squeeze = True
        else:
            self.squeeze = False
        self.window = window
        self.filters_init = activation or 'in-out'
        self.transposed = transposed
        self.bias = Variable(name='bias', shape=(size,), init='zeros') if bias else None
        self.bias = bias
        if isinstance(normalization, tuple):
            assert len(normalization) == 2
            self.normalization = Normalization(offset=normalization[0], scale=normalization[1])
            self.requires_condition = True
        elif normalization:
            self.normalization = Normalization()
            self.requires_condition = False
        else:
            self.normalization = Identity()
            self.requires_condition = False
        self.activation = Activation(activation=activation) if activation else Identity()
        self.norm_act_before = norm_act_before
        self.padding = padding
        if isinstance(stride, int):
            assert stride > 0
            self.stride = stride if len(window) == 1 else (1, stride, stride, 1)
        else:
            assert len(stride) == 2 and stride[0] > 0 and stride[1] > 0
            self.stride = (1, stride[0], stride[1], 1)
        super(Convolution, self).__init__(size=size, name=name)

    def forward(self, x, condition=None):
        if self.norm_act_before:
            if self.requires_condition:
                x = (x, condition) >> self.normalization
            else:
                x >>= self.normalization
            x >>= self.activation
        if self.transposed:
            filters_shape = self.window + (self.size, shape(x)[-1])
        else:
            filters_shape = self.window + (shape(x)[-1], self.size)
        self.filters = Variable(name='filters', shape=filters_shape, init=self.filters_init)
        if len(self.window) == 1:
            x = tf.nn.conv1d(value=x, filters=self.filters(), stride=self.stride, padding=self.padding)
        elif self.transposed:
            batch, height, width, _ = shape(x)
            x = tf.nn.conv2d_transpose(value=x, filter=self.filters(), output_shape=(batch, height * self.stride[1], width * self.stride[2], self.size), strides=self.stride, padding=self.padding)
        else:
            x = tf.nn.conv2d(input=x, filter=self.filters(), strides=self.stride, padding=self.padding)
        if self.bias:
            x = tf.nn.bias_add(value=x, bias=self.bias())
        if self.squeeze:
            x = tf.squeeze(input=x, axis=-1)
        if not self.norm_act_before:
            if self.requires_condition:
                x = (x, condition) >> self.normalization
            else:
                x >>= self.normalization
            x >>= self.activation
        if self.requires_condition:
            return x, condition
        else:
            return x


class NgramConvolution(Layer):

    def __init__(self, size, ngrams=3, padding='VALID', name=None):
        self.convolutions = []
        for ngram in range(1, ngrams + 1):  # not start with 1
            convolution = Convolution(size=size, window=ngram, normalization=False, activation='relu', padding=padding)  # norm, act?
            self.convolutions.append(convolution)
        super(NgramConvolution, self).__init__(size=size, name=name)

    def forward(self, x):
        embeddings = [x >> convolution for convolution in self.convolutions]
        # requires SAME
        # phrase_embeddings = tf.stack(values=embeddings, axis=1)
        # phrase_embeddings = tf.reduce_max(input_tensor=phrase_embeddings, axis=???)
        # two reductions, and both concat only possible with SAME
        # maybe lstm?
        return tf.concat(values=embeddings, axis=1)


# class Expand(Layer):

#     def __init__(self, size, bottleneck=False, unit=Convolution, name=None):
#         assert issubclass(unit, Layer)
#         self.bottleneck = unit(size=(size * bottleneck)) if bottleneck else Identity()
#         self.unit = unit(size=size)
#         super(Residual, self).__init__(size=size, name=name)

#     def forward(self, x):
#         fx = x >> self.bottleneck >> self.unit
#         return tf.concat(values=(x, fx), axis=(rank(x) - 1))


class Repeat(LayerStack):

    def __init__(self, layer, size, name=None, **kwargs):
        kwargs_list = [dict(size=s) for n, s in enumerate(size)]
        for name, value in kwargs.items():
            if isinstance(value, list):
                assert len(value) == len(kwargs_list)
                for n in range(len(value)):
                    kwargs_list[n][name] = value[n]
            else:
                for n in range(len(kwargs_list)):
                    kwargs_list[n][name] = value
        self.layers = list()
        for kwargs in kwargs_list:
            self.layers.append(layer(**dict(kwargs)))
        super(Repeat, self).__init__(name=name)


class ConvolutionalNet(LayerStack):

    def __init__(self, sizes, depths, pool='max', name=None):
        assert Pooling.valid(pool)
        self.layers = list()
        for m, (size, depth) in enumerate(zip(sizes, depths)):
            for n in range(depth):
                self.layers.append(Convolution(size=size))
            self.layers.append(Pooling(pool=pool))  # also only in between?
        super(LayerStack, self).__init__(name=name)


class Residual(Layer):

    def __init__(self, size, unit=Convolution, depth=2, reduction='sum', name=None):
        assert isinstance(depth, int) and depth >= 0
        self.units = []
        for n in range(depth):
            self.units.append(unit(size=size))
        super(Residual, self).__init__(size=size, name=name)

    def forward(self, x):
        res = x
        for unit in self.units:
            res >>= unit
        if shape(x) != shape(res):
            x >>= self.units[0]
        return tf.add(x=x, y=res)


class ResidualNet(LayerStack):

    # citation!

    def __init__(self, sizes, depths, layer=Convolution, transition=None, pool='max', name=None):
        assert Pooling.valid(pool)
        self.layers = list()
        for m, (size, depth) in enumerate(zip(sizes, depths)):
            if m > 0:
                self.layers.append(Pooling(pool=pool))
            # if transition:
            #     self.layers.append(transition(size=size, normalize=False, activation))
            for n in range(depth):
                if m == 0:
                    if n == 0:
                        self.layers.append(layer(size=size, normalization=False, activation=None))
                        _layer = (lambda size: layer(size=size, norm_act_before=True))  # pre activation
                else:
                    self.layers.append(Residual(size=size, unit=_layer))
        self.layers.append(Normalization())
        self.layers.append(Activation())
        super(LayerStack, self).__init__(name=name)


class Fractal(Layer):

    def __init__(self, size, unit=Convolution, depth=3, reduction='mean', name=None):
        assert isinstance(depth, int) and depth >= 0
        self.depth = depth
        self.unit = unit(size=size)
        if depth > 0:
            self.fx = Fractal(size=size, unit=unit, depth=(depth - 1))
            self.ffx = Fractal(size=size, unit=unit, depth=(depth - 1))
            self.reduction = Reduction(reduction=reduction)
        super(Fractal, self).__init__(size=size, name=name)

    def forward(self, x):
        if self.depth == 0:
            return x >> self.unit
        y = x >> self.fx >> self.ffx
        x >>= self.unit
        return (x, y) >> self.reduction


class FractalNet(LayerStack):

    def __init__(self, sizes, layer=Convolution, pool='max', name=None):
        assert Pooling.valid(pool)
        self.layers = list()
        for m, size in enumerate(sizes):
            if m > 0:
                self.layers.append(Pooling(pool=pool))
            # if transition:
            #     self.layers.append(transition(size=size, normalize=False, activation))
            self.layers.append(Fractal(size=size, unit=layer))
        super(LayerStack, self).__init__(name=name)


FullyConnected = Full = FC = Dense
