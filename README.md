# Face Generator

## Steps
### Image Alignment
```sh
$ python align_images.py raw_images/ aligned_images/
``` 
##### Parameters :
* output_size - The dimension of images for input to the model (default=1024)
* x_scale - Scaling factor for x dimension (default=1)
* y_scale - Scaling factor for y dimension (default=1)
* em_scale - Scaling factor for eye-mouth distance (default=0.1)
* use_alpha - Add an alpha channel for masking (default=False)

### Latent Vector Encoding
```sh
$ python encode_images.py --batch_size=2 --output_video=False aligned_images/ generated_images/ latent_representations/ --model_url https://drive.google.com/uc?id=1oGj5qJcbk4Mt38g1k30Bhr_awZc_beM4
```
##### Parameters :
* data_dir - Directory for storing optional models (default='data')
* mask_dir - Directory for storing optional masks (default='masks')
* load_last - Start with embeddings from directory(default='')
* dlatent_avg - Use dlatent from file specified here for truncation instead of dlatent_avg from Gs (default='')
* model_res - The dimension of images in the StyleGAN model (default=1024)
* batch_size - Batch size for generator and perceptual model (default=1)
* optimizer - Optimization algorithm used for optimizing dlatents (default='ggt')
* image_size - Size of images for perceptual model (default=256)
* resnet_image_size - Size of images for the Resnet model (default=256)
* lr - Learning rate for perceptual model (default=0.25)
* decay_rate - Decay rate for learning rate (default=0.9)
* iterations - Number of optimization steps for each batch (default=100)
* decay_steps - Decay steps for learning rate decay (as a percent of iterations) (default=4)
* early_stopping - Stop early once training stabilizes (default=True)
* early_stopping_threshold - Stop after this threshold has been reached (default=0.5)
* early_stopping_patience - Number of iterations to wait below threshold (default=10)
* load_effnet - Model to load for EfficientNet approximation of dlatents (default='data/finetuned_effnet.h5')
* load_resnet - Model to load for ResNet approximation of dlatents (default='data/finetuned_resnet.h5')
* use_preprocess_input - Call process_input() first before using feed forward net (default=True)
* use_best_loss - Output the lowest loss value found as the solution (default=True)
* average_best_loss - Do a running weighted average with the previous best dlatents found (default=0.25)
* sharpen_input - Sharpen the input images (default=True)
* use_vgg_loss - Use VGG perceptual loss; 0 to disable, > 0 to scale. (default=0.4)
* use_vgg_layer - Pick which VGG layer to use (default=9)
* use_pixel_loss - Use logcosh image pixel loss; 0 to disable, > 0 to scale (default=1.5)
* use_mssim_loss - Use MS-SIM perceptual loss; 0 to disable, > 0 to scale (default=200)
* randomize_noise - Add noise to dlatents during optimization (default=False)
* load_mask - Load segmentation masks (default=False)
* face_mask - Generate a mask for predicting only the face area (default=True)

### Face Prediction (Average Method)
```sh
$ python predict.py
``` 
##### Parameters :
* model_path - StyleGAN Model Path (default='model/model.pkl')
* latent_img1 - Latent representation image 1 path (default='latent_representations/0001_01.npy')
* latent_img2 - Latent representation image 2 path (default='latent_representations/0002_01.npy')
* w1 - Weight for image 1 (default=0.6)
* w2 - Weight for image 2 (default=0.4)
* out - Output path with extension (default='result.png')
* age - Age Coefficient where greater is getting younger (default=0)
    * -2.0 : Elder
    * -1.0 : Adult
    * 0.0 : Teen
    * 1.0 : Kid
    * 2.0 : Toddler
* gender - Gender Coefficient where 0.5 is male and -0.5 is female (default=0)

### Face Morpher
```sh
$ python morpher.py --images=<images_dir_path> --background=average --out_video=<output_path>
``` 
##### Parameters :
* src - Filepath to source image (.jpg, .jpeg, .png)
* dest - Filepath to destination image (.jpg, .jpeg, .png)
* images - Folderpath to images
* width - Custom width of the images/video [default: 500]
* height - Custom height of the images/video [default: 600]
* num - Number of morph frames [default: 20]
* fps - Number frames per second for the video [default: 10]
* out_frames - Folder path to save all image frames
* out_video - Filename to save a video
* plot - Flag to plot images to result.png [default: False]
* background - background of images to be one of (black|transparent|average) [default: black]
* version - Show version