# Animal_Classification_DL

# Animal Image Classification

This code trains a convolutional neural network (CNN) to classify images of 10 different animals.

## Data

The data consists of images of 10 animal classes:

- Butterfly
- Cat
- Chicken  
- Cow
- Dog
- Elephant
- Horse
- Sheep
- Spider
- Squirrel

The data is split into a training set and a test set. The image data is stored in the `data/animals` folder, with separate subfolders for each class. 

## Model

There are two model architectures implemented:

- `SimpleCNN`: A simple CNN with 2 convolutional layers and 2 fully connected layers
- `AdvancedCNN`: A more advanced CNN with 5 convolutional layers and 3 fully connected layers

The models are defined in `models.py`. 

## Training

The main training script is `animal_train.py`. It handles loading the data, initializing the model, defining the optimizer and loss function, training for multiple epochs, and evaluating on the test set.

Key parameters:

- `--batch-size`: Batch size for training
- `--epochs`: Number of epochs to train for
- `--log_path`: Path to save TensorBoard logs
- `--save_path`: Path to save trained model checkpoints

Use `python animal_train.py --help` to see all available arguments.

The script uses PyTorch and leverages GPU acceleration if available. Progress bars and TensorBoard logging are used to monitor training.

## Evaluation

Model accuracy on the test set is evaluated at the end of each epoch. The best performing model checkpoint is saved.

## Usage

To train a model:

```
python animal_train.py
```

This will train the `AdvancedCNN` model for 100 epochs and save checkpoints to `trained_models/animal`.

You can customize the model, hyperparameters, and output paths by modifying the commandline arguments.

## Requirements

The code requires the following packages:

- PyTorch 
- torchvision
- tensorboard
- sklearn
- tqdm

Use `pip install -r requirements.txt` to install the required packages.
 
