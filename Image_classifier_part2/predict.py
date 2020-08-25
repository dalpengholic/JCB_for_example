import numpy as np
from argparser import ArgParser
from image_processor import ImageProcessor
from load_checkpoint import LoadModel
from predictor import ImageClassifier
from check_sanity import SanityChecker

# Set a command line environemnt
parser_instance = ArgParser()
args = parser_instance.arg_parser_to_predict()
# Attributes of args : 
## args.image_path, args.checkpoint, 
## args.top_k, args.category_name, args.gpu

# Set paths for loading the data and model
# Set path for loading a single image

image_path = args.image_path
modelname = args.checkpoint

# Process the image
original_image = ImageProcessor(image_path)
processed_image = original_image.process_image()

# Load the model
model_to_train = LoadModel(modelname)
model_to_train_list = model_to_train.load_model()

# Infer for classification
guess = ImageClassifier(processed_image, model_to_train_list, topk=args.top_k, gpu=args.gpu)
guess_list = guess.predict()

if args.category_names is None:
        edited_probability_list = ['{:5.2%}'.format(round(x, 4)) for x in guess_list[0]]
        classname_list = guess_list[1]
        for i,j in zip(classname_list,edited_probability_list):
            print('The probability of the input image categorized in class no.{0} is {1}'.format(i,j))
                
# Check the sanity of the result
if args.category_names is not None:
    indexed_result = SanityChecker(guess_list, model_to_train_list, f_name=args.category_names)
    name_probability = indexed_result.index_to_flowername()
    # print the probabilities and names
    edited_probability_list = ['{:5.2%}'.format(round(x, 4)) for x in guess_list[0]]    
    flowername_list = name_probability[0]
    for i,j in zip(flowername_list,edited_probability_list):
        print('The probability of the input image categorized in the {0} is {1}'.format(i,j))

