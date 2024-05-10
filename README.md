# Python-Task
Akratech
# Image Classification using Convolutional Neural Networks (CNN)
This repository contains Python code that demonstrates building a Convolutional Neural Network (CNN) model for image classification using TensorFlow and Keras. The model is trained on the Intel Image Classification dataset, which consists of natural scenes around the world categorized into six classes: buildings, forest, glacier, mountain, sea, and street.
# Requirements
Python (>= 3.6)
TensorFlow (>= 2.0)
Keras (>= 2.0)
Matplotlib
NumPy
Pandas (for data preprocessing, if needed)
# Dataset
The dataset used for training and testing the model is the Intel Image Classification dataset available on Kaggle. It includes labeled images divided into training and test sets, organized into folders based on different image categories.

# Usage
1) Clone the Repository
git clone https://github.com/your-username/image-classification.git
cd image-classification
2) Install Dependencies
pip install -r requirements.txt
3) Data Preparation
Download the Intel Image Classification dataset and extract it.
Update the train_path and test_path variables in the Python script (image_classification.py) to point to the extracted dataset paths.
4) Training the ModelRun the following command to train the CNN model:
python image_classification.py
The script will preprocess the data, build the CNN model, train it using the training set, and evaluate its performance on the test set.
5) Model Evaluation: After training, the script will display plots showing training/validation loss and accuracy. Additionally, the model will be evaluated on both training and test datasets.
6) Inference (Prediction): To make predictions on new images, use the prediction function provided in the script. Update the testing_image variable with the path to the new image.
testing_image = "path/to/your/image.jpg"
prediction(testing_image, actual_label="glacier")
This will load the trained model and display the prediction result for the specified image.
7) Model Saving and LoadingThe trained model will be saved as Intel_images_model.h5 after training. You can load the saved model using the following code:
from tensorflow.keras.models import load_model
model = load_model("Intel_images_model.h5")
# Acknowledgments
Intel Image Classification Dataset on Kaggle
TensorFlow and Keras documentation
# Accessing other files
I have provided the model and the train and the testing folders in the name of other files, please fell free to use that.
# Requirements
I have also attached a requirements file, which includes the basic requirements for this task.
# Image Classification
The file santhosh-image-classification.ipynb has the complete code for this task.
