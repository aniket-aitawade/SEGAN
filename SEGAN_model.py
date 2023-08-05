import tensorflow as tf

def make_generator_model():
    input=tf.keras.Input(shape=[16384,1])
    z=tf.keras.Input(shape=[8,1024])
    
    conv1=tf.keras.layers.Conv1D(filters=16,kernel_size=31,strides=2,padding="same")(input)
    conv1=tf.keras.layers.PReLU()(conv1)
    
    conv2=tf.keras.layers.Conv1D(filters=32,kernel_size=31,strides=2,padding="same")(conv1)
    conv2=tf.keras.layers.PReLU()(conv2)
    
    conv3=tf.keras.layers.Conv1D(filters=32,kernel_size=31,strides=2,padding="same")(conv2)
    conv3=tf.keras.layers.PReLU()(conv3)
    
    conv4=tf.keras.layers.Conv1D(filters=64,kernel_size=31,strides=2,padding="same")(conv3)
    conv4=tf.keras.layers.PReLU()(conv4)
    
    conv5=tf.keras.layers.Conv1D(filters=64,kernel_size=31,strides=2,padding="same")(conv4)
    conv5=tf.keras.layers.PReLU()(conv5)
    
    conv6=tf.keras.layers.Conv1D(filters=128,kernel_size=31,strides=2,padding="same")(conv5)
    conv6=tf.keras.layers.PReLU()(conv6)
    
    conv7=tf.keras.layers.Conv1D(filters=128,kernel_size=31,strides=2,padding="same")(conv6)
    conv7=tf.keras.layers.PReLU()(conv7)
    
    conv8=tf.keras.layers.Conv1D(filters=256,kernel_size=31,strides=2,padding="same")(conv7)
    conv8=tf.keras.layers.PReLU()(conv8)
    
    conv9=tf.keras.layers.Conv1D(filters=256,kernel_size=31,strides=2,padding="same")(conv8)
    conv9=tf.keras.layers.PReLU()(conv9)
    
    conv10=tf.keras.layers.Conv1D(filters=512,kernel_size=31,strides=2,padding="same")(conv9)
    conv10=tf.keras.layers.PReLU()(conv10)
    
    conv11=tf.keras.layers.Conv1D(filters=1024,kernel_size=31,strides=2,padding="same")(conv10)
    conv11=tf.keras.layers.PReLU()(conv11)
    
    zc=tf.keras.layers.concatenate([conv11,z],axis=2)
    
    deconv11=tf.keras.layers.Conv1DTranspose(filters=512,kernel_size=31,strides=2,padding="same")(zc)
    deconv11=tf.keras.layers.PReLU()(deconv11)
    merge11=tf.keras.layers.concatenate([conv10,deconv11],axis=2)
    
    deconv10=tf.keras.layers.Conv1DTranspose(filters=256,kernel_size=31,strides=2,padding="same")(merge11)
    deconv10=tf.keras.layers.PReLU()(deconv10)
    merge10=tf.keras.layers.concatenate([conv9,deconv10],axis=2)
    
    deconv9=tf.keras.layers.Conv1DTranspose(filters=256,kernel_size=31,strides=2,padding="same")(merge10)
    deconv9=tf.keras.layers.PReLU()(deconv9)
    merge9=tf.keras.layers.concatenate([conv8,deconv9],axis=2)
    
    deconv8=tf.keras.layers.Conv1DTranspose(filters=128,kernel_size=31,strides=2,padding="same")(merge9)
    deconv8=tf.keras.layers.PReLU()(deconv8)
    merge8=tf.keras.layers.concatenate([conv7,deconv8],axis=2)
    
    deconv7=tf.keras.layers.Conv1DTranspose(filters=128,kernel_size=31,strides=2,padding="same")(merge8)
    deconv7=tf.keras.layers.PReLU()(deconv7)
    merge7=tf.keras.layers.concatenate([conv6,deconv7],axis=2)
    
    deconv6=tf.keras.layers.Conv1DTranspose(filters=64,kernel_size=31,strides=2,padding="same")(merge7)
    deconv6=tf.keras.layers.PReLU()(deconv6)
    merge6=tf.keras.layers.concatenate([conv5,deconv6],axis=2)
    
    deconv5=tf.keras.layers.Conv1DTranspose(filters=64,kernel_size=31,strides=2,padding="same")(merge6)
    deconv5=tf.keras.layers.PReLU()(deconv5)
    merge5=tf.keras.layers.concatenate([conv4,deconv5],axis=2)
    
    deconv4=tf.keras.layers.Conv1DTranspose(filters=32,kernel_size=31,strides=2,padding="same")(merge5)
    deconv4=tf.keras.layers.PReLU()(deconv4)
    merge4=tf.keras.layers.concatenate([conv3,deconv4],axis=2)
    
    deconv3=tf.keras.layers.Conv1DTranspose(filters=32,kernel_size=31,strides=2,padding="same")(merge4)
    deconv3=tf.keras.layers.PReLU()(deconv3)
    merge3=tf.keras.layers.concatenate([conv2,deconv3],axis=2)
    
    deconv2=tf.keras.layers.Conv1DTranspose(filters=16,kernel_size=31,strides=2,padding="same")(merge3)
    deconv2=tf.keras.layers.PReLU()(deconv2)
    merge2=tf.keras.layers.concatenate([conv1,deconv2],axis=2)
    
    deconv1=tf.keras.layers.Conv1DTranspose(filters=1,kernel_size=31,strides=2,padding="same")(merge2)
    deconv1=tf.keras.layers.PReLU()(deconv1)
    
    output=deconv1
    model=tf.keras.Model(inputs=[input,z],outputs=output)
    return model

def make_discriminator_model():
    input1=tf.keras.Input(shape=[16384,1])
    input2=tf.keras.Input(shape=[16384,1])
    
    input=tf.keras.layers.concatenate([input1,input2],axis=2)
    
    conv1=tf.keras.layers.Conv1D(filters=16,kernel_size=31,strides=2,padding="same")(input)
    conv1=tf.keras.layers.BatchNormalization()(conv1)
    conv1=tf.keras.layers.LeakyReLU(alpha=0.3)(conv1)
    
    conv2=tf.keras.layers.Conv1D(filters=32,kernel_size=31,strides=2,padding="same")(conv1)
    conv2=tf.keras.layers.BatchNormalization()(conv2)
    conv2=tf.keras.layers.LeakyReLU(alpha=0.3)(conv2)
    
    conv3=tf.keras.layers.Conv1D(filters=32,kernel_size=31,strides=2,padding="same")(conv2)
    conv3=tf.keras.layers.BatchNormalization()(conv3)
    conv3=tf.keras.layers.LeakyReLU(alpha=0.3)(conv3)
    
    conv4=tf.keras.layers.Conv1D(filters=64,kernel_size=31,strides=2,padding="same")(conv3)
    conv4=tf.keras.layers.BatchNormalization()(conv4)
    conv4=tf.keras.layers.LeakyReLU(alpha=0.3)(conv4)
    
    conv5=tf.keras.layers.Conv1D(filters=64,kernel_size=31,strides=2,padding="same")(conv4)
    conv5=tf.keras.layers.BatchNormalization()(conv5)
    conv5=tf.keras.layers.LeakyReLU(alpha=0.3)(conv5)
    
    conv6=tf.keras.layers.Conv1D(filters=128,kernel_size=31,strides=2,padding="same")(conv5)
    conv6=tf.keras.layers.BatchNormalization()(conv6)
    conv6=tf.keras.layers.LeakyReLU(alpha=0.3)(conv6)
    
    conv7=tf.keras.layers.Conv1D(filters=128,kernel_size=31,strides=2,padding="same")(conv6)
    conv7=tf.keras.layers.BatchNormalization()(conv7)
    conv7=tf.keras.layers.LeakyReLU(alpha=0.3)(conv7)
    
    conv8=tf.keras.layers.Conv1D(filters=256,kernel_size=31,strides=2,padding="same")(conv7)
    conv8=tf.keras.layers.BatchNormalization()(conv8)
    conv8=tf.keras.layers.LeakyReLU(alpha=0.3)(conv8)
    
    conv9=tf.keras.layers.Conv1D(filters=256,kernel_size=31,strides=2,padding="same")(conv8)
    conv9=tf.keras.layers.BatchNormalization()(conv9)
    conv9=tf.keras.layers.LeakyReLU(alpha=0.3)(conv9)
    
    conv10=tf.keras.layers.Conv1D(filters=512,kernel_size=31,strides=2,padding="same")(conv9)
    conv10=tf.keras.layers.BatchNormalization()(conv10)
    conv10=tf.keras.layers.LeakyReLU(alpha=0.3)(conv10)
    
    conv11=tf.keras.layers.Conv1D(filters=1024,kernel_size=31,strides=2,padding="same")(conv10)
    conv11=tf.keras.layers.BatchNormalization()(conv11)
    conv11=tf.keras.layers.LeakyReLU(alpha=0.3)(conv11)
    
    FC=tf.keras.layers.Conv1D(filters=1,kernel_size=1,strides=1,padding="same")(conv11)
    # FC=tf.keras.layers.LeakyReLU(alpha=0.3)(FC)  # NOt sure about this layer
    FC=tf.keras.layers.Flatten()(FC)
    output=tf.keras.layers.Dense(1,activation='sigmoid')(FC)
    
    model=tf.keras.Model(inputs=[input1,input2],outputs=output)    
    # model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer='adam',metrics='accuracy')

    return model

class GAN(tf.keras.Model):
    def __init__(self,generator,discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.generator=generator
        self.discriminator=discriminator
        
        
    def compile(self,g_opt,d_opt,*args,**kwargs):
        super().compile(*args,**kwargs)
        
        self.g_opt=g_opt
        self.d_opt=d_opt

        
    def train_step(self, batch):
        noisy,clean=batch
        real=clean
        
        z=tf.random.normal([tf.shape(noisy)[0],8,1024],0,1,dtype=tf.float32)
                
        with tf.GradientTape() as d_tape,tf.GradientTape() as g_tape:
            fake=self.generator([noisy,z],training=True) 
            y_hat_real=self.discriminator([noisy,real],training=True)
            y_hat_fake=self.discriminator([noisy,fake],training=True)
            # y_hat=tf.concat([y_hat_real,y_hat_fake],axis=0)
            
            # y=tf.concat([tf.zeros_like(y_hat_real),tf.ones_like(y_hat_fake)],axis=0)    
            
            # noise_real = 0.15*tf.random.uniform(tf.shape(y_hat_real))
            # noise_fake = -0.15*tf.random.uniform(tf.shape(y_hat_fake))
            # y += tf.concat([noise_real, noise_fake], axis=0)
            
            total_d_loss=tf.reduce_mean(tf.math.squared_difference(y_hat_real,1.))+tf.reduce_mean(tf.math.squared_difference(y_hat_fake,0.))
            
            total_g_loss=tf.reduce_mean(tf.math.squared_difference(y_hat_fake,1.))+100*tf.reduce_mean(tf.abs(tf.subtract(fake,real)))
            
            
            # total_d_loss=self.d_loss(y,y_hat)  
            
            # total_g_loss=self.g_loss(tf.zeros_like(y_hat_fake),y_hat_fake)
            
        d_grad=d_tape.gradient(total_d_loss,self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad,self.discriminator.trainable_variables))
        
        g_grad=g_tape.gradient(total_g_loss,self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad,self.generator.trainable_variables))
    
        
        return {"total d loss":total_d_loss,"total g loss":total_g_loss}
