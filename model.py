import tensorflow as tf

# To know more about layers, go to - https://keras.io/api/layers/
# 1D Convolution Layers - https://keras.io/api/layers/convolution_layers/convolution1d/
# 1D MaxPooling Layers - https://keras.io/api/layers/pooling_layers/max_pooling1d/
# 1D UpSampling Layers - https://keras.io/api/layers/reshaping_layers/up_sampling1d/
# Concatenate Layer - https://keras.io/api/layers/merging_layers/concatenate/
# Optimizer - https://keras.io/api/optimizers/
# Losses - https://keras.io/api/losses/
# Metrics - https://keras.io/api/metrics/

# Layer and Model subclasing - https://www.youtube.com/watch?v=WcZ_1IAH_nM&ab_channel=AladdinPersson

class DS_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(DS_block, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="LeakyReLU")
        self.downsample = tf.keras.layers.MaxPooling1D(pool_size=2)
    
    def call(self, input_tensor, training=False):
        x = self.conv1d(input_tensor, training=training)
        y = self.downsample(x, training=training)
        return x, y

class US_block(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super(US_block, self).__init__()
        self.upsample = tf.keras.layers.UpSampling1D(size=2)
        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation="LeakyReLU")
        self.concat = tf.keras.layers.Concatenate(axis=2)
    
    def call(self, input_tensor, extra_tensor, training=False):
        x = self.upsample(input_tensor, training=training)
        trim = extra_tensor.shape[1] - x.shape[1]
        if trim%2==0:   cropping = (trim//2, trim//2)
        else:   cropping = (trim//2, trim//2+1)
        tmp_crop = tf.keras.layers.Cropping1D(cropping=cropping)(extra_tensor)
        x = self.concat([x, tmp_crop])
        x = self.conv1d(x, training=training)
        return x

class Downsample_Model(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, L):
        super(Downsample_Model, self).__init__()
        self.DS = [DS_block(filters=filters*(i+1), kernel_size=kernel_size) for i in range(L)]

    def call(self, input_tensor, training=False):
        history = []
        y = input_tensor
        for layer in self.DS:
            x, y = layer(y, training=training)
            history.append(x)
        return y, history

class Upsample_Model(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, L):
        super(Upsample_Model, self).__init__()
        self.US = [US_block(filters=filters*(i+1), kernel_size=kernel_size) for i in range(L-1, -1, -1)]

    def call(self, input_tensor, extra_tensors, training=False):
        x = input_tensor
        i = len(self.US) - 1
        for layer in self.US:
            x = layer(input_tensor=x, extra_tensor=extra_tensors[i], training=training)
            i = i - 1
        return x

class WaveUNet(tf.keras.Model):
    def __init__(self, kernel_size, filters, L):
        super(WaveUNet, self).__init__(name='')
        self.Downsample = Downsample_Model(filters=filters, kernel_size=kernel_size, L=L)
        self.conv1d_DS = tf.keras.layers.Conv1D(filters=filters*(L+1), kernel_size=kernel_size, strides=1, activation="LeakyReLU")
        self.Upsample = Upsample_Model(filters=filters, kernel_size=kernel_size, L=L)
        self.conv1d_output = tf.keras.layers.Conv1D(filters=2, kernel_size=kernel_size, strides=1, activation="tanh")

    def call(self, input_tensor, training=False):
        x, history = self.Downsample(input_tensor, training=training)
        x = self.conv1d_DS(x, training=training)
        x = self.Upsample(input_tensor=x, extra_tensors=history, training=training)
        x = self.conv1d_output(x, training=training)
        diff = input_tensor.shape[1] - x.shape[1] 
        if diff%2==0: 
            pad = (diff//2, diff//2)
        else:
            pad = (diff//2, diff//2+1)
        x = tf.keras.layers.ZeroPadding1D(padding=pad)(x)
        return x

    def model(self, data_size):
        x = tf.keras.Input(shape=(data_size, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
