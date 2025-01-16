# Deep-learning-project
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def build_cnn_model(input_shape, num_classes):
    """
    Build a CNN model for image classification.
    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of target classes.
    Returns:
        tensorflow.keras.Model: Compiled CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history):
    """
    Plot the training and validation accuracy and loss.
    Args:
        history (tensorflow.keras.callbacks.History): History object from model training.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    # Data preparation
    data_dir = "data"  # Replace with your dataset directory
    image_size = (64, 64)
    batch_size = 32

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )

    # Model building
    input_shape = image_size + (3,)
    num_classes = len(train_generator.class_indices)
    model = build_cnn_model(input_shape, num_classes)

    # Model training
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    # Visualize results
    plot_training_history(history)

    # Save the model
    model.save("image_classification_model.h5")
    print("Model saved as 'image_classification_model.h5'")
