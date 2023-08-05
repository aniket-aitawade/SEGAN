import tensorflow as tf
import os

from preprocessing import *
from SEGAN_model import *

devices=tf.config.list_physical_devices('GPU')
for i in range(len(devices)):
  tf.config.experimental.set_memory_growth(devices[i],True)
  
# Define Path of noisy and clean audios in directory
directory='D:\Dataset\noisy','D:\Dataset\clean'
audios=os.listdir(directory[0])
dataset=Custom_dataloader(directory=directory,audios=audios,batch=40,frame_length=16384,hop_length=16384,preemphasis_coefficient=0.95,target_sr=16000)
dataloader=tf.data.Dataset.from_generator(generator=lambda: (dataset[i] for i in range(len(dataset))),
                                          output_signature=((tf.TensorSpec(shape=(None,16384,1), dtype=tf.float32),
                                                             tf.TensorSpec(shape=(None,16384,1), dtype=tf.float32))))
training_data=dataloader.prefetch(tf.data.AUTOTUNE)
training_data=training_data.cache()

g_opt=tf.keras.optimizers.RMSprop(learning_rate=0.0002)
d_opt=tf.keras.optimizers.RMSprop(learning_rate=0.0002)

generator=make_generator_model()
discriminator=make_discriminator_model()
SEGAN_model=GAN(generator,discriminator)
SEGAN_model.compile(g_opt,d_opt)

SEGAN_model.fit(training_data,batch_size=32,epochs=86)
# tf.keras.Model.save_weights(SEGAN_model,'1306_SEGAN_weights.ckpt')