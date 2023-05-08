# Car Image Classification Project

This project is about car image classification using Tensorflow and Keras.

## Test data

The test data folder structure of all the classes of images should look like below:

```
data
 |
 |-test
     |
     |-bmw serie 1
         |
         |-back5.jpg
         |-back9.jpg
         |.....
     |-chevrolet spark
         |
         |-back50.jpg
         |-back52.jpg
         |.....
```

## Running test script

To run the test script, follow the step below:

1. Install the reuiqred packages by running `pip install -r requirement.txt`
2. Run the test.py file by executing the following command:

```
python test.py --model_path './model/' --image_folder_path './data/test/[classes]' --output_file_path './output.txt'
```

_The [classes] should be change to the classes name of the car.
For example './data/test/bwm serie 1'_

3. The script will load the saved model and use it to make predictions on the test data.
4. The results will be saved to a output.txt file in the same directory.

Note: Before running the train script, please make sure to place your training and validate data in the "./data/train" and the "./data/val" folder. In order to reduce the size of the project file, the dataset in this project folder contains only test image.
