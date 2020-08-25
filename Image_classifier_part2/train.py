from argparser import ArgParser
import os
from load_data import DataForTorch
from building_classifier import BuildModel
from training_classifier import TrainModel
from save_checkpoint import SaveModel
from load_checkpoint import LoadModel

# Set a command line environemnt
parser_instance = ArgParser()
args = parser_instance.arg_parser_to_train()
# Attributes of args : 
## args.data_dir, args.arch, args.learning_rate
## args.hidden_units, args.gpu, args.epochs, args.save_dir


# Set paths for loading the data
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Set paths for loading the model
modelname = args.save_dir
exist = os.path.isfile('/home/workspace/ImageClassifier/'+modelname)

# Prepare the data
data_for_torch= DataForTorch(train_dir,valid_dir,test_dir)
data_for_model_list = data_for_torch.prepare_data()

if exist:
    # Load the model
    model_to_train = LoadModel(modelname, learning_rate=args.learning_rate)
    model_to_train_list = model_to_train.load_model()

else :
    # Build a model
    model_to_train = BuildModel(architecture=args.arch, learning_rate=args.learning_rate, hidden_units=args.hidden_units)
    model_to_train_list = model_to_train.build_model()

# Train the model
modelfortorch = TrainModel(data_for_model_list, model_to_train_list, gpu=args.gpu, epochs=args.epochs)
trained_model_list = modelfortorch.train_model()

# Save the model
model_to_save = SaveModel(trained_model_list, savepath=args.save_dir)
model_to_save.save_classifier()

