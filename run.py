import numpy as np
import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image
import os
#import tensorflow as tf

#Image Alignment
os.system("python align_images.py raw_images/ aligned_images/")

#Latent Vector Generation
os.system("python encode_images.py --batch_size=2 --output_video=False aligned_images/ generated_images/ latent_representations/ --model_url model/model.pkl")

# load the StyleGAN model into Colab
#URL_FFHQ = 'https://drive.google.com/uc?id=1oGj5qJcbk4Mt38g1k30Bhr_awZc_beM4'
tflib.init_tf()
#with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
with open('model/model.pkl','rb') as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)
# load the latents
s1 = np.load('latent_representations/0001_01.npy')
s2 = np.load('latent_representations/0002_01.npy')
s1 = np.expand_dims(s1,axis=0)
s2 = np.expand_dims(s2,axis=0)
# combine the latents somehow... let's try an average:
savg = 0.2*s1 + 0.8*s2
# run the generator network to render the latents:
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
images = Gs_network.components.synthesis.run(savg, randomize_noise=False, **synthesis_kwargs)
(PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS)).save("result.png")
#tf.reset_default_graph()
