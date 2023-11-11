from model_utils.data_generators import DirectoryGenerator
import tensorflow as tf
import keras_tuner as kt
import numpy as np

input_shape   = (72,72,3)
output_shape  = 4
BATCH_SIZE = 64

def build_model(hp):
  inputs = tf.keras.Input(shape=input_shape)
  x = inputs
  for i in range(hp.Int('conv_blocks', 3, 5, default=3)):
    filters = hp.Int(f'filters_{str(i)}', 32, 256, step=32)
    for _ in range(2):
      x = tf.keras.layers.Convolution2D(
        filters, kernel_size=(3, 3), padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.ReLU()(x)
    if hp.Choice(f'pooling_{str(i)}', ['avg', 'max']) == 'max':
      x = tf.keras.layers.MaxPool2D()(x)
    else:
      x = tf.keras.layers.AvgPool2D()(x)
  x = tf.keras.layers.GlobalAvgPool2D()(x)
  x = tf.keras.layers.Dense(
      hp.Int('hidden_size', 30, 100, step=10, default=50),
      activation='relu')(x)
  x = tf.keras.layers.Dropout(
      hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)
  outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)

  model = tf.keras.Model(inputs, outputs)
  model.compile(
    optimizer=tf.keras.optimizers.Adam(
      hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
    loss='categorical_crossentropy',
    metrics=['accuracy'])
  return model

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    hyperband_iterations=2,
    directory=r"H:\Mash_kers_tuner",
    project_name="mash_search")

# tuner = kt.RandomSearch(
#     hypermodel=build_model,
#     objective="val_accuracy",
#     max_trials=500,
#     executions_per_trial=5,
#     overwrite=False,
#     directory=r"model_files\random_search",
#     project_name="mash_search"
# )

train_datasets = [r"train"]
val_datasets   = [r"validation"]
train_dataset_weights = None # must be integers, and same length as <train_datasets>
val_datasets_weights  = None # must be integers, and same length as <val_datasets>

train_generator = DirectoryGenerator(train_datasets, input_shape, dataset_weights=train_dataset_weights, class_weights=[1,1,1,1])
val_generator   = DirectoryGenerator(val_datasets,   input_shape, dataset_weights=val_datasets_weights)

train = train_generator.get_all_data()
test = val_generator.get_all_data()

batch_size = 64

indices = np.random.permutation(len(train[0]))
shuffled_images = train[0][indices]
shuffled_labels = train[1][indices]
batch_images = shuffled_images[:batch_size]
batch_labels = shuffled_labels[:batch_size]

tuner.search(batch_images,batch_labels,epochs=30,
             validation_data=(test[0],test[1]),
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]