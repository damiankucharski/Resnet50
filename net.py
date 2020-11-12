from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model

class Resnet50:

    def __init__(self, input_shape = (64, 64, 3), classes = 10):
        """
        Implementation of the ResNet50 architecture based on paper https://arxiv.org/abs/1512.03385:

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        """
        

        X_input = Input(input_shape)

        X = ZeroPadding2D((3, 3))(X_input)

        X = Conv2D(64, (7, 7), strides = (2, 2))(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        X = self.convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
        X = self.identity_block(X, 3, [64, 64, 256])
        X = self.identity_block(X, 3, [64, 64, 256])

        X = self.convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])
        X = self.identity_block(X, 3, [128, 128, 512])

        X = self.convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])
        X = self.identity_block(X, 3, [256, 256, 1024])

        X = self.convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
        X = self.identity_block(X, 3, [512, 512, 2048])
        X = self.identity_block(X, 3, [512, 512, 2048])

        X = AveragePooling2D()(X)
        
        X = Flatten()(X)

        if classes > 2:
            X = Dense(classes, activation='softmax')(X)
        if classes == 2:
            X = Dense(1, activation='sigmoid')(X)

        model = Model(inputs = X_input, outputs = X)
        self.model = model
        

    def compile(self, **kwargs):
        self.model.compile(**kwargs)


    def identity_block(self, X, f, filters):
        """
        Implementation of the identity block
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        
        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """
        
        F1, F2, F3 = filters
        
        X_shortcut = X
        
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)
        

        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same')(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)

        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
        X = BatchNormalization(axis = 3)(X)

        X = Add()([X_shortcut, X]) 
        X = Activation('relu')(X)
        
        
        return X


    def convolutional_block(self, X, f, filters, s = 2):
        """
        Implementation of the convolutional block used for matching input output shapes if they are not the same
        
        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        s -- Integer, specifying the stride to be used
        
        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """
    

        F1, F2, F3 = filters

        X_shortcut = X

        X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid')(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)

        X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same')(X)
        X = BatchNormalization(axis = 3)(X)
        X = Activation('relu')(X)


        X = Conv2D(F3, (1, 1), strides = (1,1), padding = 'valid')(X)
        X = BatchNormalization(axis = 3)(X)

        
        X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), padding = 'valid')(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)
        

        
        return X

