#!/usr/bin/env python3
"""
train.py
Skeleton training script for Sankatmochan (MobileNetV2 fine-tuning).
Usage examples:
    python train.py --data_dir /content/dataset --epochs 10 --batch_size 32 --output_dir ../backend/ml_model
"""

import argparse
import os
import datetime
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

def build_model(input_shape=(224,224,3), dropout_rate=0.3):
    # Base model (ImageNet weights)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False   # freeze base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def get_generators(data_dir, img_size=(224,224), batch_size=32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.15,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        fill_mode="nearest"
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator

def make_callbacks(output_dir, monitor="val_loss"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logdir = os.path.join(output_dir, "tensorboard", timestamp)
    os.makedirs(tb_logdir, exist_ok=True)

    es = EarlyStopping(monitor=monitor, patience=6, restore_best_weights=True, verbose=1)
    ckpt = ModelCheckpoint(os.path.join(output_dir, "best_model.h5"),
                           monitor=monitor, save_best_only=True, verbose=1)
    tb = TensorBoard(log_dir=tb_logdir, histogram_freq=1)
    rlrop = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, verbose=1)

    return [es, ckpt, tb, rlrop], tb_logdir

def convert_to_tflite(keras_model_path, tflite_output_path, quantize=False):
    print("Converting to TFLite...")
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        # Post-training float16 quantization (smaller, faster)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    print("TFLite saved to:", tflite_output_path)

def main(args):
    # Generators
    train_gen, val_gen = get_generators(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)

    # Build model
    model = build_model(input_shape=(args.img_size, args.img_size, 3), dropout_rate=args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Callbacks
    callbacks, tb_logdir = make_callbacks(args.output_dir, monitor="val_loss")

    # Training loop
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        workers=4,
        use_multiprocessing=False
    )

    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.h5")
    model.save(final_model_path)
    print("Saved final Keras model to:", final_model_path)

    # TFLite conversion (optional)
    tflite_path = os.path.join(args.output_dir, "model.tflite")
    convert_to_tflite(final_model_path, tflite_path, quantize=args.quantize)

    # Simple validation metrics printout
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    print(f"Validation loss: {val_loss:.4f}  acc: {val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/content/dataset", help="path to dataset (train/val)")
    parser.add_argument("--output_dir", type=str, default="../backend/ml_model", help="where to save models and logs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--quantize", action="store_true", help="apply float16 quantization for TFLite")
    args = parser.parse_args()
    main(args)
