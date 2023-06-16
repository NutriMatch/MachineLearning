# NutriMatch-MachineLearning
## Convolutional Neural Network 
Convolutional Neural Network or CNN is a type of neural network often used for computer vision. It utilizes three types of layers: a convolutional layer to extract features, a pooling layer to downsample the feature maps, and a fully connected layer to classify the extracted features.

## Transfer Learning 
Transfer learning is a machine learning technique where a pre-trained model, trained on a large dataset for a specific task, is utilized as a starting point for a new but related task. By leveraging the learned features and representations from the pre-trained model, transfer learning enables faster and more efficient training, improves performance on tasks with limited data, and contributes to advancements in various domains such as computer vision.

## Datasets 
Datasets were collected with the browser extension ImageAssistant Batch Image Downloader which utilizes web scrape, then carefully evaluated manually to ensure the quality. Consists of 10 types of food, namely rice, noodles, fish, chicken, egg (boiled or sunny side up), broccoli, bread, orange, tofu, and tempeh. Each food is representing one of the essential nutrients for our body.

## Training 
### Datasets used 
specify input shape of (416,416) 

### Preprocess 
Using ImageDataGenerator to do augmentation. 

### Architecture
Used transfer learning approach with YOLOv3. Choosing the layer from YOLOv3 to be the last layer and the input for the additional layer. Set sigmoid as the activation function for multi-label classification. 

### Saving Model
The model was saved as .h5 format, ready to be deployed.

## Evaluate
Evaluation was done in Load_Test_Model.ipynb using f1, precision, and recall as the evaluation metrics.
