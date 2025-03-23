import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np
import cv2

def create_skin_classifier():
    # Load ResNet50V2 with pre-trained weights
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Unfreeze more layers for fine-tuning
    for layer in base_model.layers[:-50]:  # Fine-tune more layers
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    
    # Add spatial attention
    attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attention])
    
    x = GlobalAveragePooling2D()(x)
    
    # Deeper network with careful regularization
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer with temperature scaling
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Use a lower learning rate
    optimizer = Adam(learning_rate=5e-5)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )
    
    return model

def train_model(train_dir, validation_dir):
    model = create_skin_classifier()
    
    # More aggressive data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,  # Faces shouldn't be flipped vertically
        fill_mode='reflect',
        brightness_range=[0.7, 1.3],
        shear_range=0.2,
        channel_shift_range=20,  # Add color variation
        validation_split=0.15
    )
    
    valid_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet.preprocess_input
    )
    
    # Smaller batch size for better generalization
    batch_size = 8
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Calculate class weights
    total_samples = train_generator.samples
    class_counts = [len(os.listdir(os.path.join(train_dir, c))) for c in ['dry', 'normal', 'oily']]
    max_count = max(class_counts)
    class_weights = {i: max_count/count for i, count in enumerate(class_counts)}
    
    # Enhanced callbacks
    callbacks = [
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=12,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')  # Log training metrics
    ]
    
    # Print model summary
    model.summary()
    
    print("\nStarting model training...")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print("\nClass distribution:")
    for i, count in enumerate(['dry', 'normal', 'oily']):
        print(f"{count}: {class_counts[i]} images (weight: {class_weights[i]:.2f})")
    
    # Train with more epochs
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=100,  # Increased epochs with better early stopping
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        workers=1
    )
    
    # Save the final model
    model.save('skin_classifier_model.h5')
    
    return model, history

if __name__ == "__main__":
    # Set memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass
    
    dataset_path = "dataset"
    train_dir = os.path.join(dataset_path, 'train')
    validation_dir = os.path.join(dataset_path, 'validation')
    
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        print("Error: Dataset directories not found!")
        print(f"Please ensure the following directories exist:")
        print(f"- {train_dir}")
        print(f"- {validation_dir}")
        exit(1)
    
    model, history = train_model(train_dir, validation_dir)
    print("Model trained and saved as 'skin_classifier_model.h5'") 