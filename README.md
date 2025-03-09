# Ophanim: seeing Soviet aircraft on declassified cold war satellite images

This is the code featured in [this video] (https://www.youtube.com/watch?v=T3IwER3yXs4&t=585s)

![About the limit of my blender abilities](https://github.com/lambdaBoost/Ophanim/blob/main/docs/ophanim.png )

Everything here must be prefixed with the disclaimer that this code is VERY simple for a neural net. There is nothing fancy like MLFlow integration, feature engineering pipelines or advanced model tuning. It was written over the course of a couple of evenings with the sole purpose of finding some aircraft. The model returns many false positives, and with a little extra work, it would be made much more accurate.

## Prerequisites
This should run on a base installation of Conda. However, at the time of writing, Tensorflow was having some kind of integration issue with the new 50-series GPUs. There are 2 options to deal with this:

**Option 1 - probably best for those stumbling across this from youtube and with less experience of data science**

The train_cnn script has a line near the top to disable the GPU. Just uncomment that line and run. It will take longer to train but should work

**Option 2 - Use Nvidia's official docker container to run the code**

This is what I ended up doing. The instructions to pull and run the container are [here] (https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html)

I ended up putting all the necessary training data in a single folder (called 'M15-CNN') and mounting that to the container. So, to run it, I'd use the following command. The training script could then be run as normal. The resulting model was copied back to the host machine.

```
sudo docker run --gpus all -ti --rm -v ./M15-CNN:/M15-CNN--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/tensorflow:25.01-tf2-py3
```

To fix this, I ended up carrying all the training runs out  

## Building a dataset
Bear in mind, this repo is basically just a few scripts. You just need to run them in the correct order and edit the global variables at the top.

There are 2 scripts to build a dataset:

* image_processer.py takes whole satellite images from the SOURCE_DIR folder (specified near the top of the script) and splits them into smaller blocks. The size of the blocks is specified in other global variables at the top of the script. The TEST_PROP variable specifies the proportion of images that will be saved to a 'test' folder. The rest go to a 'train' folder. For inference data (ie unlabelled images) TEST_PROP should be set to zero.

* create_positives.py takes a folder of sliced images and adds 'fake' aircraft to the images to create positive cases. The input and outputs folders are specified at the top of the script, along with a few paramaters which define the randomisation of the aircraft masks.

* If both the above scripts are run successfully, you will have a training set of positive and negative images. I have included some examples of fake PZL M-15 templates in the 'synthetic data' folder. You could use any template you wish here.


## Training a classifier
The train_cnn.py script handles this. Just point the keras dataset to the directory where your training set images live (line 16 I think) and a convolutional neural net will be trained up. At the time of writing, the script just outputs the model to the root directory and its name is specified right at the end of the code (you can change it if you want)...I should probably tidy that up a bit.

## Predicting on new images
Assuming you ran the 'process_images) script earlier on your unlabelled imaghes. You will have a seperate folder of 128x128 unlabelled image slices. infer_all_parallel.py will use our trained model to identify images with a high liklihood of containing an aircrft. Change the first few lines to point to the correct directory of images and the model we just trained. The OUTPUT_PREDICTIONS and OUTPUT_FILENAMES will be the names of our prediction data which is saved to disk as pickle files. Saving the predictions like this allows us to make use of keras's rapid inference functionality whilst saving the results.

I wrote a seperate script to actually process the predictions. process_predictions.py takes those 2 pickle files we just generated and fetches the images identified most likely as matches. The THRESHOLD parameter determines the confidence threshold at which to declare a prediction as positive. 
I ended up training seperate models for Ukraione and North CAucus. Even then, I had to do multiple models for the seperate 'regions' within these areas. The detection threshold returned far too many false positives when used in some regions so I had to experiment to get a sensibly low number of positive images to manually sift through.

The images identified as positive will be copied to the folder specified in the DEST variable at the top of the script. I just stuck the X-Y coordinates (well, actually the coordinates divided by 64.....let's not get into it) of each image slice from the 'big image' in the filename so you can find where it came from. A janky solution but it worked well enough for my purposes.

I ended up with about 3000 falkse positives from a dataset of 25 million images. Again, for the purpose of my video, that was completely fine. Just don't go using this for anything serious I guess.

## Final thoughts and improvements
As I said, this isn't anything fancy but it works. If you wanted to use this for something more serious and persistent it should probably be packaged into a proper library and run in conjunction with MLFlow for proper model versioning etc. I do enough of that for my day job, I'm not doing it here.

As for the model itself, there are endless things we could do to improve it. Improved preprocessing of the images would be the first step. A greater variation of templates and maybe some feature extraction would be the first thing I'd do. The actual model is a very simple CNN...I'm not sure a more complex onw would actually help that much but it might. I've had good success with U-nets for this kind of thing before. Of you could try something that isn't a neural net at all but be warned...it may take ages to train.

