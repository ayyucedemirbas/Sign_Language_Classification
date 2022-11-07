# Sign_Language_Classification


We build and train a classifier model on [ASL_Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) from scratch. However, it takes a lot of time. Thus, we fine-tune ResNet50 model which pretrained on imagenet. We simply freeze all the layers except the last four layers and train those four layers instead of training the whole model.


The first model looks like this:

    inputs= keras.Input(shape=(224,224,3))
    x = data_augmentation(inputs)
    x = layers.experimental.preprocessing.Rescaling(1./255)(x)


    #Block: 1
    x = Conv2D(256, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    residual = x
    x = Conv2D(256, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    residual = Conv2D(256,1, strides=2)(residual)
    x = layers.add([x, residual])
    x = Dropout(0.3)(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)


    #Block: 2
    x = Conv2D(64, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)


    x = Flatten(name='flatten')(x)
    x = Dense(units=32, activation='relu')(x)
    outputs = Dense(units=29, activation='softmax')(x)
    model= keras.Model(inputs=inputs, outputs=outputs)


We get the pretrained ResNet50 model, and build our model as follows:

    resnet_base= tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    resnet_base.trainable = True
    
    for layer in resnet_base.layers[: -4]:
        layer.trainable=False
    
    inputs= keras.Input(shape=(224,224,3))
    x = data_augmentation(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    x = resnet_base(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(units=29, activation='softmax')(x)
    model= keras.Model(inputs=inputs, outputs=outputs)
  
Since we train only four layers, the training doesn't take as much time as the one above.
