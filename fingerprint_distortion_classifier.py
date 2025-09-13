#!/usr/bin/env python3
"""
Fingerprint Image Classification using Transfer Learning with VGG16
This script builds and trains a model to classify fingerprint images into:
- High Quality
- Scratched
- Blurry
"""

# 1. SET UP THE ENVIRONMENT
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.metrics import TopKCategoricalAccuracy

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Check TensorFlow version and GPU availability
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

# Configuration parameters
IMG_SIZE = (224, 224)  # VGG16 input size
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.0001
DATASET_PATH = "fpds"
MODEL_SAVE_PATH = "fingerprint_classifier_model"

# 2. LOAD THE DATA
print("Setting up data generators...")

# Data augmentation for training set to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to [0,1]
    rotation_range=20,           # Random rotation up to 20 degrees
    width_shift_range=0.2,       # Random horizontal shift
    height_shift_range=0.2,      # Random vertical shift
    shear_range=0.2,            # Shear transformation
    zoom_range=0.2,             # Random zoom
    horizontal_flip=True,        # Random horizontal flip
    fill_mode='nearest',         # Fill strategy for new pixels
    validation_split=0.2         # Use 20% for validation
)

# Validation data generator (only rescaling, no augmentation)
validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',    # Multi-class classification
    subset='training',           # Use training subset
    shuffle=True,
    seed=42
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',         # Use validation subset
    shuffle=False,               # Don't shuffle validation data
    seed=42
)

# Get class information
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"Found {train_generator.samples} training images")
print(f"Found {validation_generator.samples} validation images")
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")

# 3. BUILD THE MODEL
print("Building the  ..")

# Load pre-trained VGG16 model without top layers
base_model = VGG16(
    weights='imagenet',          # Use ImageNet pre-trained weights
    include_top=False,           # Exclude the final classification layer
    input_shape=(*IMG_SIZE, 3)   # Input shape for RGB images
)

# Freeze the base model weights
base_model.trainable = False

print(f"Base model has {len(base_model.layers)} layers")
print("Base model weights are frozen")

# Add more capacity for 6-class classification
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.6),         # Increased dropout
    layers.Dense(512, activation='relu', name='dense_hidden_1'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', name='dense_hidden_2'), 
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax', name='predictions')
])

# Display model architecture
model.summary()

# 4. COMPILE THE MODEL
print("Compiling the model...")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
        TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

# 5. SET UP CALLBACKS
# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,                  # Stop if no improvement for 5 epochs
    restore_best_weights=True,   # Restore best weights
    verbose=1
)

# Model checkpoint to save best model during training
checkpoint = ModelCheckpoint(
    f"{MODEL_SAVE_PATH}_best.h5",
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

callbacks = [early_stopping, checkpoint]

# 6. TRAIN THE MODEL
print("Starting training...")

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = validation_generator.samples // BATCH_SIZE

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# 7. EVALUATE THE MODEL
print("Evaluating the model...")

# Evaluate on validation data
val_results = model.evaluate(
    validation_generator,
    steps=validation_steps,
    verbose=1
)

# Print all metrics with their names
for name, value in zip(model.metrics_names, val_results):
    print(f"{name}: {value:.4f}")

# 8. SAVE THE MODEL
print("Saving the model...")

# Save in SavedModel format (recommended for production)
model.save(MODEL_SAVE_PATH)
print(f"Model saved in SavedModel format at: {MODEL_SAVE_PATH}")

# Also save in H5 format for compatibility
h5_path = f"{MODEL_SAVE_PATH}.h5"
model.save(h5_path)
print(f"Model saved in H5 format at: {h5_path}")

# Save class names for later use
class_names_file = f"{MODEL_SAVE_PATH}_classes.txt"
with open(class_names_file, 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")
print(f"Class names saved to: {class_names_file}")

# 9. PLOT TRAINING HISTORY
print("Plotting training history...")

def plot_training_history(history):
    """Plot training and validation accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{MODEL_SAVE_PATH}_training_history.png", dpi=300, bbox_inches='tight')
    plt.show()

# Plot the training history
plot_training_history(history)

# 10. FINE-TUNING (OPTIONAL)
print("\n" + "="*50)
print("OPTIONAL: Fine-tuning the model")
print("="*50)

fine_tune = input("Do you want to fine-tune the model? (y/n): ").lower() == 'y'

if fine_tune:
    print("Fine-tuning the model...")
    
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = len(base_model.layers) - 10  # Last 10 layers
    
    # Freeze all layers except the last few
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE/10),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_2_accuracy']
    )
    
    print(f"Fine-tuning from layer {fine_tune_at} onwards")
    
    # Train for a few more epochs
    fine_tune_epochs = 5
    
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=fine_tune_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate fine-tuned model
    val_loss, val_accuracy, val_top2_acc, val_top3_acc = model.evaluate(
        validation_generator,
        steps=validation_steps,
        verbose=1
    )
    
    print(f"Fine-tuned Validation Loss: {val_loss:.4f}")
    print(f"Fine-tuned Validation Accuracy: {val_accuracy:.4f}")
    print(f"Fine-tuned Validation Top-2 Accuracy: {val_top2_acc:.4f}")
    print(f"Fine-tuned Validation Top-3 Accuracy: {val_top3_acc:.4f}")
    
    # Save the fine-tuned model
    fine_tuned_path = f"{MODEL_SAVE_PATH}_fine_tuned"
    model.save(fine_tuned_path)
    model.save(f"{fine_tuned_path}.h5")
    print(f"Fine-tuned model saved at: {fine_tuned_path}")

print("\n" + "="*50)
print("TRAINING COMPLETED!")
print("="*50)
print(f"Models saved:")
print(f"- Base model: {MODEL_SAVE_PATH}")
print(f"- Base model (H5): {h5_path}")
print(f"- Class names: {class_names_file}")
if fine_tune:
    print(f"- Fine-tuned model: {fine_tuned_path}")

# 11. EXAMPLE PREDICTION FUNCTION
def predict_image(model_path, image_path, class_names):
    """
    Example function to make predictions on a single image
    This can be used in your FastAPI app
    """
    # Load the saved model
    model = keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img = keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    return predicted_class, confidence

print("\nExample usage for prediction:")
print(f"predicted_class, confidence = predict_image('{MODEL_SAVE_PATH}', 'path/to/image.jpg', {class_names})")