# The Documentation from Machine Learning Path

## About dataset
in the beginning we looked for a dataset from Kaggle with [35 classes](https://www.kaggle.com/datasets/arizbw/traditional-food-knowledge-of-indonesia?select=train.csv), but after training data and testing the model, there were problems during the live test. After that we tried to find a dataset from Google and the results were bad. by discussing with the team, we agreed to reduce the classes in the dataset and increase the amount of data in each class, and finally only 15 classes with the dataset link [here](https://github.com/citradisi/ML-Models/tree/master/food-tfk-images/data).

The dataset has training data and test data totaling 2988 images, of which training data has 597 batch sizes and test data 150 batch sizes. each class has ± 200 images then where 80% training data and 20% test data. we use [image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory) to create the dataset and [VGG16](https://keras.io/api/applications/vgg/) for models.

**DISCLAIMER** 

Copyright images from google images or the uploader of the images. Our team only collected and gathered the images and then labeling for training model object detection.

## Workflow
![ML](https://ia802708.us.archive.org/27/items/ml-work_v2/ml-work.PNG)

we build the model with VGG16
