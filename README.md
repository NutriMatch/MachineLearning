# NutriMatch-MachineLearning
## Convolutional Neural Network
Convolutional Neural Network or CNN is a type of neural network often used for computer vision. It utilizes three types of layers: a convolutional layer to extract features, a pooling layer to downsample the feature maps, and a fully connected layer to classify the extracted features.
Collaborative Filtering illustration
Collaborative Filtering

## Datasets
datasets were collected with browser extension ImageAssistant Batch Image Downloader which utilize web scrape, then carefully evaluated manually to ensure the quality. Consists of 10 types of food, namely rice, noodles, fish, chicken, egg (boiled or sunny side up), broccoli, bread, orange, tofu, and tempeh. Each food is representing one of the essential nutrients for our body.

## Transfer Learning
Transfer learning is a machine learning technique where a pre-trained model, trained on a large dataset for a specific task, is utilized as a starting point for a new but related task. By leveraging the learned features and representations from the pre-trained model, transfer learning enables faster and more efficient training, improves performance on tasks with limited data, and contributes to advancements in various domains such as computer vision.


Training
The model is a rankings model consists of 2 submodels; retreival model and ranker model.

Retrieval Model
Retrieval model is used to map the keahlian or the volunteer's mission category and volunteer_id into embeddings. Therefore, a TensorFlow embedding algorithm is used for each of the 2 variables. Then at the final layers, a sequential model is used which consists of: - Dense(units=16, activation='relu') layer - Dense(units=32, activation='relu') layer - Dense(1) layer

Rankings Model
The rankings model is then used as the second submodels to rank the possible recommendation to the user based on the most to the least recommended.

Dataset used
Dummydataset dataset is used to try and train the model. For the deployment a different dummy dataset which is already on the database will be used.

Results
- `mean_squared_error: 11,65%`
- `loos : 1,35%`
- `regularization_loss: 0%`
- `total_loss: 1,35%`
Deployment
The model architecture then deployed to backend service / google cloud for then the model will get the data and process it and finally send the recommendation to the application

train.py
The train file is used to constantly train the model so that it can adapt to the updated dataset from the user. After training the model with the updated data, the file will then save the model to the google cloud storage, which then be used by the predict file to send the recommendation to the Android application

predict.py
Predict is used to get the model that is saved for predictions. The file will get the updated data based on a specific volunteer id. The predict will return the recommendations prediction in a JSON format.

Prerequisites
Function dependencies used in this project:

keras==2.7.0
numpy==1.20.3
pandas==1.3.4
tensorflow==2.7.0
tensorflow_recommenders==0.6.0
