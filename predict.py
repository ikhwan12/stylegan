# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:14:27 2020

@author: ikhwa
"""

import numpy as np
import dnnlib.tflib as tflib
import pickle
import PIL.Image
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Face prediction', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model_path', default='model/model.pkl', help='StyleGAN Model Path')
	parser.add_argument('--latent_img1', default='latent_representations/0001_01.npy', help='Latent representation image 1 path')
	parser.add_argument('--latent_img2', default='latent_representations/0002_01.npy', help='Latent representation image 2 path')
	parser.add_argument('--w1', default=0.6, help='Weight for image 1', type=float)
	parser.add_argument('--w2', default=0.4, help='Weight for image 2', type=float)
	parser.add_argument('--age', default=0, help='Age Coefficient where greater is getting younger', type=float)
	parser.add_argument('--gender', default=0, help='Gender Coefficient where 0.5 is male and -0.5 is female', type=float)
	parser.add_argument('--out', default='result.png', help='Output path with extension')
	args, other_args = parser.parse_known_args()
	
	MODEL_PATH = args.model_path
	gender_coeff = args.gender
	age_coeff = args.age
	
	tflib.init_tf()
	#with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
	with open(MODEL_PATH,'rb') as f:
	    generator_network, discriminator_network, Gs_network = pickle.load(f)
	gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
	age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
	# load the latents
	s1 = np.load(args.latent_img1)
	s2 = np.load(args.latent_img2)
	s1 = np.expand_dims(s1,axis=0)
	s2 = np.expand_dims(s2,axis=0)
	# combine the latents somehow... let's try an average:
	w1 = args.w1
	w2 = args.w2
	if w1 + w2 > 1 :
		w1 = 0.6
		w2 = 0.4
	savg = w1*s1 + w2*s2
	#Gender Transformation
	gender_vector = savg.copy()
	gender_vector[:8] = (savg + gender_coeff*gender_direction)[:8]
	#Age Transformation
	age_vector = gender_vector.copy()
	age_vector[:8] = (gender_vector + age_coeff*age_direction)[:8]
	# run the generator network to render the latents:
	synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
	images = Gs_network.components.synthesis.run(age_vector, randomize_noise=False, **synthesis_kwargs)
	(PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS)).save(args.out)