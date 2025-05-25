import os
import sys
import glob
import argparse 
import time
import numpy as np
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from src import *
from tensorflow import keras
from src import data_lowlight
from src.loss import *
from src.model import DCE_x
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate, Conv2D
from PIL import Image

def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.JPG")
    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, lowlight_images_path, batch_size):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 512
        self.data_list = self.train_list
        self.batch_size = batch_size
        self.n_channels = 3
        print("Total training examples:", len(self.train_list))	

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.data_list[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(indexes)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        X = np.empty((self.batch_size, self.size, self.size, self.n_channels), dtype=np.float32)  # Ensure float32 dtype
        for i, data_lowlight_path in enumerate(indexes):
            data_lowlight = Image.open(data_lowlight_path)
            data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
            data_lowlight = np.asarray(data_lowlight) / 255.0
            X[i] = data_lowlight.astype(np.float32)
        return tf.convert_to_tensor(X, dtype=tf.float32)

def progress(epoch, trained_sample ,total_sample, bar_length=25, total_loss=0, message=""):
    if total_sample == 0:
        print("Error: total_sample is zero!")
        return
    percent = float(trained_sample) / total_sample
    hashes = '#' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write("\rEpoch {0}: [{1}] {2}%  ----- Loss: {3}".format(epoch, hashes + spaces, int(round(percent * 100)), float(total_loss)) + message)
    sys.stdout.flush()

def eval(model):
    for data_lowlight_path in glob.glob("test/" + "*.jpg"):
        original_img = Image.open(data_lowlight_path)
        original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])
        original_img = original_img.resize((512,512), Image.LANCZOS) 
        original_img = (np.asarray(original_img)/255.0).astype(np.float32)
        img_lowlight = Image.open(data_lowlight_path)
        img_lowlight = img_lowlight.resize((512,512), Image.LANCZOS)
        img_lowlight = (np.asarray(img_lowlight)/255.0).astype(np.float32) 
        img_lowlight = np.expand_dims(img_lowlight, 0)
        A = model.predict(img_lowlight) 
        r1, r2, r3, r4, r5, r6, r7, r8 = A[:,:,:,:3], A[:,:,:,3:6], A[:,:,:,6:9], A[:,:,:,9:12], A[:,:,:,12:15], A[:,:,:,15:18], A[:,:,:,18:21], A[:,:,:,21:24]
        x = original_img + r1 * (K.pow(original_img,2)-original_img)
        x = x + r2 * (tf.pow(x,2)-x)
        x = x + r3 * (tf.pow(x,2)-x)
        enhanced_image_1 = x + r4*(K.pow(x,2)-x)
        x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
        x = x + r6*(tf.pow(x,2)-x)	
        x = x + r7*(tf.pow(x,2)-x)
        enhance_image = x + r8*(K.pow(x,2)-x)
        enhance_image = tf.cast((enhance_image[0,:,:,:] * 255), dtype=np.uint8)
        enhance_image = Image.fromarray(enhance_image.numpy())
        enhance_image = enhance_image.resize(original_size, Image.LANCZOS)
        enhance_image.save(data_lowlight_path.replace('.jpg', '_rs.jpg'))

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    train_dataset = data_lowlight.DataGenerator(config.lowlight_images_path, config.train_batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr)
    input_img = Input(shape=(512, 512, 3), dtype=tf.float32)  # Ensure input dtype is float32
    conv1 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(input_img)
    conv2 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(conv2)
    conv4 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(conv3)
    int_con1 = Concatenate(axis=-1)([conv4, conv3])
    conv5 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(int_con1)
    int_con2 = Concatenate(axis=-1)([conv5, conv2])
    conv6 = Conv2D(32, (3, 3), strides=(1,1), activation='relu', padding='same')(int_con2)
    int_con3 = Concatenate(axis=-1)([conv6, conv1])
    x_r = Conv2D(24, (3,3), strides=(1,1), activation='tanh', padding='same')(int_con3)
    model = Model(inputs=input_img, outputs=x_r)
    min_loss = 10000.0

    print("Start training ...")
    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_dataset):
            img_lowlight = tf.cast(img_lowlight, tf.float32)  # Ensure img_lowlight is float32
            with tf.GradientTape() as tape:
                A = model(img_lowlight)
                # Split into r1 to r8, each cast to float32
                r1, r2, r3, r4, r5, r6, r7, r8 = [tf.cast(A[:, :, :, i*3:(i+1)*3], tf.float32) for i in range(8)]
                # Cast img_lowlight to float32 before computations
                x = tf.cast(img_lowlight, tf.float32) + r1 * (tf.pow(tf.cast(img_lowlight, tf.float32), 2) - tf.cast(img_lowlight, tf.float32))
                x = x + r2 * (tf.pow(x, 2) - x)
                x = x + r3 * (tf.pow(x, 2) - x)
                enhanced_image_1 = x + r4 * (tf.pow(x, 2) - x)
                x = enhanced_image_1 + r5 * (tf.pow(enhanced_image_1, 2) - enhanced_image_1)		
                x = x + r6 * (tf.pow(x, 2) - x)	
                x = x + r7 * (tf.pow(x, 2) - x)
                enhance_image = x + r8 * (tf.pow(x, 2) - x)
                
                # Loss calculations
                loss_TV = 200 * L_TV(A)
                loss_spa = tf.reduce_mean(L_spa(enhance_image, img_lowlight))
                loss_col = 5 * tf.reduce_mean(L_color(enhance_image))
                loss_exp = 10 * tf.reduce_mean(L_exp(enhance_image, mean_val=0.6))
                total_loss = loss_TV + loss_spa + loss_col + loss_exp

            # Apply gradients
            grads = tape.gradient(total_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            # Progress update (optional)
            progress(epoch+1, (iteration+1), len(train_dataset), total_loss=total_loss)

        print(f"\nEnd of epoch {epoch+1}\n")

        # Save checkpoint after each epoch
        if (epoch + 1) % config.checkpoint_iter == 0:
            checkpoint_path = os.path.join(config.checkpoints_folder, f"model_epoch_{epoch+1}.h5")
            model.save(checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowlight_images_path', type=str, default="data/lowlight/")
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint_iter', type=int, default=1000)
    parser.add_argument('--checkpoints_folder', type=str, default="checkpoints/")
    config = parser.parse_args()
    if not os.path.exists(config.checkpoints_folder):
        os.makedirs(config.checkpoints_folder)
    train(config)
