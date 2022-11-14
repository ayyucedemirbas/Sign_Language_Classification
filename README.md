# Sign_Language_Classification


We build and train a classifier model on [ASL_Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). We get 0.99333 accuracy on the validation set.

The  model looks like this:

![image](https://user-images.githubusercontent.com/8023150/201570312-858d3fce-1f31-4657-bfa1-98bbf229ac01.png)

    inputs= keras.Input(shape=(100,100,3))
    #x = data_augmentation(inputs)
    x = layers.experimental.preprocessing.Rescaling(1./255)(inputs)


    #Block: 1
    x = Conv2D(64, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    residual = x
    x = SeparableConv2D(64, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding = 'same')(x)
    residual = Conv2D(64,1, strides=2)(residual)
    x = layers.add([x, residual])
    x = Dropout(0.3)(x)
    x = SeparableConv2D(32, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    residual = x
    x = SeparableConv2D(32, kernel_size=(3, 3), activation='relu',padding='SAME')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    residual = Conv2D(32,1, strides=2)(residual)
    x = layers.add([x, residual])
    x = Dropout(0.3)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(units=32, activation='relu')(x)
    outputs = Dense(units=29, activation='softmax')(x)
    model= keras.Model(inputs=inputs, outputs=outputs)
    
<img width="386" alt="image" src="https://user-images.githubusercontent.com/8023150/200340107-c0f50229-3b58-469a-8021-cdf7edbdf729.png">
<img width="386" alt="image" src="https://user-images.githubusercontent.com/8023150/200340207-f0727382-520e-4791-8583-67d88ca92a6e.png">



We also fine-tune ResNet50 model which pretrained on imagenet. We simply freeze all the layers except the last four layers and train those four layers instead of training the whole model.

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
  
