# Data
(Not uploaded to github)
Data is stored in the data direcotry. rawdata contains the raw dataset before preprocessing, croppeddata contains the files after cropping, and preprocessdata contains the final preprocesssed data with contrast normalization and padding. Under each directory contains test, train, and val directories that each contain 8 directories corresponding to the 8 classes, AMD, CNV, CSR, DME, DR, DRUSEN, MH, and NORMAL.

# Preprocessing:
## cropwhiteborder.py:
Removes white border from classes DME, DRUSEN, and NORMAL.
Images with white borders have varying dimensions, so component labeling is necessary. Finds connected regions of white pixels on each of the 4 edges with component labeling. Then, filters to keep regions that only touch the image's border and at least 1% of the image. For each edge, cropping stops if less than 30% of the whitepixels in consecutive rows (for top and bottom edges)/columns (for left and right edges) are border connected pixels. Crops the edge with the most area of white pixels and loops until there are no significant white borders left.

## cropblackborder.py
Removes black borders from classes DR and MH. Simple center crop that ouputs as a 650x420 image. Does not need the checks the white border classes do, as the black borders are not as extreme as the white borders and share the same dimensions. 

## contrast_padding.py
Applies contrast and brightness normalziation across all images with target mean of 128 and standard deviation of 45.
After contrast and brightness normalization, take all cropped images and insert black padding to the top and bottom of the images to make the dimension ratio 1:1. The amount of padding at the top and bottom is randomized for the training scans to minimize bias for classes that have the same dimension throughout.

# Training
We trained our CNN model by using a DenseNet model, which is pretrained on ImageNet. We resized all scans to 224x224, applied a 3 channel grayscale, and normalized the channels with mean of .485, .456, .406 and std of .229, .224, .225, as those are the ImageNet dataset's mean and std values. These preprocessing steps are to calibrate for the DenseNet model specifically. For the training dataset, we made 50% of the images randomly flip horizontally and a random rotation of 15 degrees to decrease bias. 
Along with using DenseNet as our pretrained model, we used the AdamW optimizer and set the learning rate at 1e-4 and the weight decay at 1e-4. We chose AdamW over Adam becuase it handles weight decay more effctively, which prevents overfitting. We set these weights at these values because we are using a pretrained model with weights that are already trained on the much larger ImageNet database, so we only want small updates to the weights. 
Chooses the best model based on the F1 score.

# Evaluation
## Gradcam
In order to see what our model was actually basing its classifications on, we generated Gard-CAM heatmaps to visualize which regions of the scan our model focuses on. With this technique, we were able to see that some classes were prone to have bias, as the model would focus on parts of the scan that are not related to eye disease. We overlay the heatmap onto the original image.