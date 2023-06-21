import tensorflow as tf

class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filters , kernels ,strides=(1,1),padding='valid',name=None):
        super(ConvolutionBlock, self  ).__init__(name=name)
        self.conv = tf.keras.layers.Conv2D(filters, kernels, strides=strides, padding=padding)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        # self.scale = tf.keras.layers.Activation('sigmoid')
        self.relu = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batch_norm(x)
        # x = self.scale(x)
        output = self.relu(x)
        return output

class InceptionV1Block(tf.keras.layers.Layer):
    def __init__(self, filters , name =None):
        super(InceptionV1Block, self).__init__(name=name)
        self.conv1x1 = ConvolutionBlock(filters[0], (1, 1))
        self.conv3x3_reduce = ConvolutionBlock(filters[1], (1, 1))
        self.conv3x3 = ConvolutionBlock(filters[2], (3, 3), padding='same')
        self.conv5x5_reduce = ConvolutionBlock(filters[3], (1, 1))
        self.conv5x5 = ConvolutionBlock(filters[4], (5, 5), padding='same')
        self.maxpool = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.conv1x1_reduce = ConvolutionBlock(filters[5], (1, 1))

    def call(self, inputs):
        conv1x1 = self.conv1x1(inputs)
        conv3x3_reduce = self.conv3x3_reduce(inputs)
        conv3x3 = self.conv3x3(conv3x3_reduce)
        conv5x5_reduce = self.conv5x5_reduce(inputs)
        conv5x5 = self.conv5x5(conv5x5_reduce)
        maxpool = self.maxpool(inputs)
        conv1x1_reduce = self.conv1x1_reduce(maxpool)

        output = tf.concat([conv1x1, conv3x3, conv5x5, conv1x1_reduce], axis=-1)
        return output
    

class SEBlock(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.squeeze = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(num_channels // self.reduction_ratio, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_channels, activation='sigmoid')

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.keras.layers.Reshape((1, 1, -1))(x)
        output = inputs * x
        return output

class SEPBlock(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16,name =None):
        super(SEPBlock, self).__init__(name=name)
        self.scale = SEBlock(reduction_ratio)
        self.statistical_module = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        self.pruning_block = tf.keras.layers.Multiply()

    def call(self, inputs):
        x = self.scale(inputs)
        x_stat = self.statistical_module(x)
        x_pruned = self.pruning_block([x, x_stat])
        output = inputs + x_pruned
        return output
def get_model(outputs:int=2)->tf.keras.Model:
    Inputs = tf.keras.Input(shape=(224,224,3))
    x = ConvolutionBlock(64,(7,7),(2,2),padding='same' , name = 'Conv1')(Inputs)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = ConvolutionBlock(64,(1,1),strides=(1,1),padding='same',name='Conv2a')(x)
    x = ConvolutionBlock(192,(3,3),strides=(1,1),padding='same',name='Conv2b')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = InceptionV1Block([64,96,128,16,32,32],name='Inception3a')(x) # 3a
    x = SEPBlock(name='SEP3a')(x)  # 3a
    x = InceptionV1Block([128,128,192,32,96,64],name='Inception3b')(x) # 3b
    x = SEPBlock(name='SEP3b')(x) # 3b
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = InceptionV1Block([192,96,208,6,48,64],name='Inception4a')(x) # 4a
    x = SEPBlock(name='SEP4a')(x) # 4a
    x = InceptionV1Block([160,112,224,24,64,64],name='Inception4b')(x) # 4b
    _4bx = SEPBlock(name='SEP4b')(x) # 4b
    x = InceptionV1Block([128,128,256,24,64,64],name='Inception4c')(_4bx) # 4c
    _4cx = SEPBlock(name='SEP4c')(x) # 4c
    x = InceptionV1Block([112,144,288,32,64,64],name='Inception4d')(_4cx) # 4d
    x = SEPBlock(name='SEP4d')(x) # 4d
    x = InceptionV1Block([256,160,320,32,128,128],name='Inception4e')(x) # 4e
    x = SEPBlock(name='SEP4e')(x) # 4e
    x = tf.concat([x,_4bx,_4cx],axis=-1)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(outputs,activation='softmax',name='classifier')(x)
    model = tf.keras.Model(inputs=Inputs, outputs=x)
    model.summary()

# Unit test 
# model =get_model(2)