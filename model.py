import tensorflow as tf

# To know more about layers, go to - https://keras.io/api/layers/
# 1D Convolution Layers - https://keras.io/api/layers/convolution_layers/convolution1d/
# 1D Cropping Layers - https://keras.io/api/layers/reshaping_layers/cropping1d/
# Concatenate Layer - https://keras.io/api/layers/merging_layers/concatenate/
# Optimizer - https://keras.io/api/optimizers/
# Losses - https://keras.io/api/losses/
# Metrics - https://keras.io/api/metrics/

# Layer and Model subclasing - https://www.youtube.com/watch?v=WcZ_1IAH_nM&ab_channel=AladdinPersson

def upsample(x):
    x = tf.expand_dims(x, axis=1)
    x = tf.image.resize(x, [1, x.get_shape().as_list()[2] * 2 - 1], method=tf.image.ResizeMethod.BILINEAR)
    x = tf.squeeze(x, axis=1)
    return x

def crop_and_concat(t1, t2):
    trim = t1.shape[1] - t2.shape[1]
    if trim%2==0:   
        cropping = (trim//2, trim//2)
    else:   
        cropping = (trim//2, trim//2+1)
    tmp_crop = tf.keras.layers.Cropping1D(cropping=cropping)(t1)
    return tf.keras.layers.Concatenate(axis=2)([t2, tmp_crop])
        

class DS_block(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super(DS_block, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1)
        self.norm = tf.keras.layers.BatchNormalization()
    
    def call(self, input_tensor, training=False):
        x = self.conv1d(input_tensor, training=training)
        x = self.norm(x, training=training)
        y = x[:,::2,:]
        return x, y

class US_block(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size):
        super(US_block, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=1)
        self.norm = tf.keras.layers.BatchNormalization()
    
    def call(self, input_tensor, extra_tensor, training=False):
        x = upsample(input_tensor)
        x = crop_and_concat(extra_tensor, x)
        x = self.conv1d(x, training=training)
        x = self.norm(x, training=training)
        return x

class WaveUNet(tf.keras.Model):

    def __init__(self, kernel_size, filters, L):
        super(WaveUNet, self).__init__(name='')
        assert len(filters)==2, "Give two separate filters for the DS and US blocks, received " + str(len(filters))
        assert len(kernel_size)==2, "Give two separate kernel sizes for the DS and US blocks, received " + str(len(kernel_size))
        self.L = L
        self.DS = [DS_block(filters=filters[0]*(i+1), kernel_size=kernel_size[0]) for i in range(self.L)]
        self.US = [US_block(filters=filters[1]*(i+1), kernel_size=kernel_size[1]) for i in range(self.L)]
        self.conv1d_DS = tf.keras.layers.Conv1D(filters=filters[0]*(L+1), kernel_size=kernel_size[0], strides=1)
        self.conv1d_US = tf.keras.layers.Conv1D(filters=2, kernel_size=1, strides=1, activation="tanh")
        self.norm_input = tf.keras.layers.BatchNormalization()
        self.norm_DS = tf.keras.layers.BatchNormalization()
        self.norm_US = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.norm_input(input_tensor, training=training)
        history = []
        for i in range(self.L):
            q, x = self.DS[i](x, training=training)
            history.append(q)
        
        x = self.conv1d_DS(x, training=training)
        x = self.norm_DS(x, training=training)

        for i in range(self.L-1, -1, -1):
            x = self.US[i](x, history[i], training=training)

        x = crop_and_concat(input_tensor, x)
        x = self.conv1d_US(x, training=training)
        x = self.norm_US(x, training=training)

        return x

    def model(self, data_size):
        x = tf.keras.Input(shape=(data_size, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

model = WaveUNet([15, 5], [24, 24], 5)
model.model(8000).summary()

