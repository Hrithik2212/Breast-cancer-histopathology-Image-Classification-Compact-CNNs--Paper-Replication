from model import get_model
import os
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    # Set the base directory containing your image dataset
    base_dir = '/content/drive/MyDrive/BACH-dataset'

    # Define the target directories for train, test, and validation sets
    train_dir = '/content/drive/MyDrive/BACH-dataset/train_dir'
    test_dir = '/content/drive/MyDrive/BACH-dataset/test_dir'
    val_dir = '/content/drive/MyDrive/BACH-dataset/val_dir'

    # Get the list of subdirectories (classes) in the base directory
    classes = os.listdir(base_dir)

    # Iterate over each class and split the images into train, test, and validation sets
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        images = os.listdir(class_dir)

        # Split the images into train, test, and validation sets
        train_images, test_val_images = train_test_split(images, test_size=0.2, random_state=42)
        test_images, val_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

        # Create the target directories if they don't exist
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # Move the images to their respective directories
        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_dir, class_name, image)
            shutil.move(src, dst)

        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(test_dir, class_name, image)
            shutil.move(src, dst)

        for image in val_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(val_dir, class_name, image)
            shutil.move(src, dst)
    def walk_through_dir(dir_path):
        for dirpath , dirnames , filenames in os.walk(dir_path):
            print(f'There are {len(dirnames)} directories and {len(filenames)} in {dirpath}')
        walk_through_dir('/content/drive/MyDrive/BACH-dataset')

    # Base directory containing the image data

    # Define the data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Set the image directory and other parameters for the generators
    train_dir = '/content/drive/MyDrive/BACH-dataset/train_dir'
    test_dir = '/content/drive/MyDrive/BACH-dataset/test_dir'
    val_dir = '/content/drive/MyDrive/BACH-dataset/val_dir'
    target_size = (224, 224)
    batch_size = 8

    # Create the train, test, and validation generators
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
    )
    model = get_model(4)

    print("\n[INFO] Ready to train. Training is starting!\n")

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)   # limitation to stop somewhere early

    hist = model.fit(train_generator, validation_data=val_generator, epochs=25 , callbacks=[callback])

    model.save('SqueezeExcitationPruneInception.h5')