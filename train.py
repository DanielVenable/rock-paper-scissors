import keras

def create_model() -> keras.Sequential:
    """ Creates a model from MobileNetV2 with a few added layers. """

    # Use MobileNet with imagenet weights as a base model
    base_net = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model so it doesn't update during training.
    base_net.trainable = False

    input_layer = keras.layers.Input(shape=(224, 224, 3))

    x = keras.layers.Rescaling(1./255)(input_layer)
    x = keras.layers.RandomFlip()(x)
    x = keras.layers.RandomRotation(1)(x)
    x = base_net(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)

    output_layer = keras.layers.Dense(3, activation='softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train(model: keras.Model):
    training_set = keras.preprocessing.image_dataset_from_directory('data', image_size=(224, 224))
    model.fit(training_set, epochs=20)

def save(model: keras.Model):
    model.save('model.keras')

model = create_model()
train(model)
save(model)
print('Model saved!')