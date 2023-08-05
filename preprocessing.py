import tensorflow as tf
import librosa
import os
import numpy as np

class Custom_dataloader(tf.keras.utils.Sequence):
  def __init__(self,directory,audios,batch,frame_length,hop_length,preemphasis_coefficient=0.95,target_sr=16000,):
    self.noisy=[os.path.join(directory[0],audios) for audios in audios]
    self.clean=[os.path.join(directory[1],audios) for audios in audios]
    self.batch=batch
    self.frame_length=frame_length
    self.hop_length=hop_length
    self.preemphasis_coefficient=preemphasis_coefficient
    self.target_sr=target_sr
    
  def __len__(self):
    return int(len(self.noisy)/self.batch)
    
  def audio_loader(self,noisy,clean):
    noisy_audio,sr1=librosa.load(noisy)
    noisy_audio=librosa.resample(noisy_audio,orig_sr=sr1,target_sr=self.target_sr)
    clean_audio,sr2=librosa.load(clean)
    clean_audio=librosa.resample(clean_audio,orig_sr=sr2,target_sr=self.target_sr)
    return noisy_audio,clean_audio

  def preemphasis(self,noisy,clean):
    noisy_preemphasis=np.concatenate([[noisy[0]],noisy[1:]-self.preemphasis_coefficient * noisy[:-1]],axis=0)
    clean_preemphasis=np.concatenate([[clean[0]],clean[1:]-self.preemphasis_coefficient * clean[:-1]],axis=0)
    return noisy_preemphasis,clean_preemphasis

  def framing(self,noisy,clean):
    noisy_frames=librosa.util.frame(noisy,frame_length=self.frame_length,hop_length=self.hop_length,axis=0)
    clean_frames=librosa.util.frame(clean,frame_length=self.frame_length,hop_length=self.hop_length,axis=0)
    return noisy_frames,clean_frames

  def __getitem__(self,idx):
    noisy_=self.noisy[idx*self.batch:idx*self.batch+self.batch]
    clean_=self.clean[idx*self.batch:idx*self.batch+self.batch]

    noisy=np.zeros([1,self.frame_length])
    clean=np.zeros([1,self.frame_length])

    for i in range(self.batch):
      noisy_audio,clean_audio=self.audio_loader(noisy_[i],clean_[i])        
      noisy_preemphasis,clean_preemphasis=self.preemphasis(noisy_audio,clean_audio)
      noisy_frames,clean_frames=self.framing(noisy_preemphasis,clean_preemphasis)

      noisy=np.concatenate([noisy,noisy_frames])
      clean=np.concatenate([clean,clean_frames])
    
    noisy=np.expand_dims(noisy[1:],axis=2)
    clean=np.expand_dims(clean[1:],axis=2)

    return tf.constant(noisy,dtype=tf.float32),tf.constant(clean,dtype=tf.float32)
  