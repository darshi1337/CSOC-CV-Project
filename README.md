# Chess Piece Recognition and Data Augmentation Report

## Initial Plan and Implementation of Virtual Chessboard using Pygame

The initial plan was to train a model capable of recognizing chess pieces and the chessboard and then utilize Reinforcement Learning (RL) to predict the next move. However, as no suitable RL libraries were available for this specific task, the first step was to create a virtual chessboard using Pygame. The Python code utilizes Pygame version 2.5.0 to implement a virtual chessboard on the computer. The chessboard is represented as a dictionary where each square is identified by its position (e.g., '1A' for the top-left corner) and the corresponding chess piece ('r' for a black rook, 'p' for a black pawn, etc.). Chess pieces are displayed using Unicode characters, which are stored in a separate dictionary called 'PIECES'. The 'draw_chessboard()' function is responsible for rendering the chessboard, iterating through each square, and displaying the appropriate chess pieces based on the 'chessboard' dictionary.

## Data Augmentation for Chess Piece Dataset

Since there was a limited number of test cases available, data augmentation was implemented to increase the dataset's diversity. To achieve this, the 'imgaug' library was used. The 'augment_image()' function accepts a list of image paths, reads each image using 'imageio', and applies various augmentations using the 'iaa.Sequential' method from 'imgaug'. The augmented images are then saved to the specified output directory. Additionally, the 'augment_dataset()' function augments all images for each piece type in the given dataset directory, creating additional augmented images to enhance the dataset's variability.

## Model Training and Evaluation

To perform chess piece recognition, a Convolutional Neural Network (CNN) model was employed. The model was trained using different combinations of hyperparameters, including the number of convolutional layers, nodes in each layer, and dense layers. The augmented dataset was used for training the model. The architecture of the CNN model consisted of convolutional layers followed by ReLU activation functions and max-pooling layers. Dense layers with ReLU activation and dropout regularization were incorporated to prevent overfitting. The final layer utilized a softmax activation function for multi-class classification. The model was compiled with categorical cross-entropy loss and optimized using the Adam optimizer. Training was conducted using the 'fit()' method with both training and validation data generators. The loop iterated over various hyperparameter combinations to train and evaluate multiple models.

## Time and Resource Consumption

The training process consumed unexpectedly long durations, exceeding 36 hours. This extended time frame could be attributed to the extensive combinations of hyperparameters and the augmentation of the dataset. The computationally intensive nature of the training process posed challenges, especially on standard personal computers.

## Overall Accuracy Calculation

After evaluating the accuracy of the trained models, an overall accuracy score was computed. Models with 128 nodes in the convolutional layers were assigned double importance while calculating the overall accuracy.

## Model Architecture and Summary

The model summary for one of the trained models with 128 nodes in the convolutional layers is presented. The architecture of the model consists of multiple convolutional layers, each followed by activation functions (ReLU) and max-pooling layers. Dense layers with ReLU activation and dropout regularization are employed to enhance model performance.

## Chess Piece Recognition and Image Capture using Webcam

The trained model is utilized to predict chess pieces from images captured through the webcam. OpenCV is employed to capture webcam images. The 'predict_chess_piece()' function processes the image and returns the predicted class (chess piece). The predicted class is then displayed on the image using OpenCV's 'putText()' function. The processed image is converted to RGB format and displayed using matplotlib. The loop continues until the 'q' key is pressed, allowing for continuous recognition of chess pieces from the webcam feed.
