import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns

# Set GPU memory consumption growth to avoid memory errors and optimize resource usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

root_directory = r'C:\Users\Madu\Desktop\Shilpi\shilpi\colon_image_sets'

# Define paths to image directories
training_path = os.path.join(root_directory, 'train')
validation_path = os.path.join(root_directory, 'val')
testing_path = os.path.join(root_directory, 'test')

# Check the number of images in each folder
training_size = sum([len(files) for r, d, files in os.walk(training_path)])
validation_size = sum([len(files) for r, d, files in os.walk(validation_path)])
testing_size = sum([len(files) for r, d, files in os.walk(testing_path)])

print(f'Training size: {training_size}')
print(f'Validation size: {validation_size}')
print(f'Testing size: {testing_size}')

# Define data generators for the training, validation, and testing sets
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

testing_datagen = ImageDataGenerator(rescale=1./255)

# Use generators to load images from directories
training_batches = training_datagen.flow_from_directory(
    training_path,
    target_size=(224, 224),
    batch_size=32
)
training_labels = list(training_batches.class_indices.keys())

validation_batches = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(224, 224),
    batch_size=32
)

testing_batches = testing_datagen.flow_from_directory(
    testing_path,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False
)
testing_labels = list(testing_batches.class_indices.keys())

# Print the shapes of the datasets
print('Training shape:', training_batches[0][0].shape)
print('Validation shape:', validation_batches[0][0].shape)
print('Testing shape:',  testing_batches[0][0].shape)


# Plot some images from the training dataset and save a copy
plt.figure(figsize=(10, 10))
for i in range(15):
    images, labels = next(training_batches)
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(images[0])
    plt.title(f"{training_labels[np.argmax(labels[0])]} ({np.argmax(labels[0])})")  # Show the class name and index of each image sample
    plt.axis("on")
plt.savefig('sample_images.png')
plt.show()

# Load the pre-trained model
base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Freeze the convolutional layers
for layer in base_model.layers[:]:
    layer.trainable = False

# Add a new classifier different from the built-in classifiers of VGG16
last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output

# Flatten the classifier input from the last layer of the VGG16 model
x = Flatten()(last_output)

# Add a fully connected layer of 256 units and batchnorm, dropout, and softmax layers
x = Dense(256, activation='relu', name='Startline')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', name='lastline')(x)

# Define the new model
new_model = Model(inputs=base_model.input, outputs=x)
new_model.summary()

# Compile the model with categorical cross-entropy loss, Adam optimizer, and learning rate of 1e-4
new_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

# Define callbacks
early_stop = EarlyStopping(patience=3)
model_checkpoint = ModelCheckpoint('best_model.hdf5', verbose=2, save_best_only=True)

# Fit the model with 1000 epochs
history = new_model.fit(training_batches, epochs=1000, steps_per_epoch=128, validation_data=validation_batches, callbacks=[early_stop, model_checkpoint])

# Load the best model from the checkpoint
best_model = tf.keras.models.load_model('best_model.hdf5')

# Evaluate the model on the training, validation, and testing sets
train_loss, train_accuracy = best_model.evaluate(training_batches)
val_loss, val_accuracy = best_model.evaluate(validation_batches)
test_loss, test_accuracy = best_model.evaluate(testing_batches)

# Print the evaluation results
print('Training Loss:', train_loss)
print('Training Accuracy:', train_accuracy)
print('Validation Loss:', val_loss)
print('Validation Accuracy:', val_accuracy)
print('Testing Loss:', test_loss)
print('Testing Accuracy:', test_accuracy)

# Plot the model accuracy evaluation and save a copy of the plot
plt.plot(history.history['accuracy'], color='teal')
plt.plot(history.history['val_accuracy'], color='orange')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.show()

# Plot the model loss evaluation and save a copy of the plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_loss.png')
plt.show()

# Get the predicted labels for the testing set
predictions = best_model.predict(testing_batches)
predicted_labels = np.argmax(predictions, axis=1)

# Get the true labels for the testing set
true_labels = testing_batches.classes

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.show()

# Calculate precision, recall, F1 score, and support for each class
class_report = classification_report(true_labels, predicted_labels, target_names=testing_labels)
print("Classification Report:")
print(class_report)

# Calculate the false positive rate and true positive rate for ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predictions[:, 1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
plt.show()


