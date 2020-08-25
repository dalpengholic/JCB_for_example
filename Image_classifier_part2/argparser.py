class ArgParser:
    

    def arg_parser_to_train(self):
        import argparse
        parser = argparse.ArgumentParser(description='Argparse for training')
        parser.add_argument('data_dir', type=str, default='./flowers', action='store', help='Set the directory of data for training')
        parser.add_argument('--arch', type=str, default='vgg16', action='store', choices=['vgg13', 'vgg16', 'densenet121'] ,help='Set a base model for training')
        parser.add_argument('--learning_rate', type=float, default=0.002, action='store', help='Set a learning rate(float type) for the model')
        parser.add_argument('--hidden_units', type=int, default=4096, action='store', help='Set the number of Hidden units(int type) for the model')
        parser.add_argument('--epochs', type=int, default=1, action='store', help='Set the number of epochs(int type) for the model')
        parser.add_argument('--gpu', action='store_true', default='gpu', help='GPU on and off for training')
        parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', action='store', help='Set a directory for saving a checkpoint')
        # Parse args
        args = parser.parse_args()
        
        return args

    def arg_parser_to_predict(self):
        import argparse
        parser = argparse.ArgumentParser(description='Argparse for prediction')
        parser.add_argument('image_path', type=str, default='./flowers/test/10/image_07090.jpg', action='store', help='Input a path of an image')
        parser.add_argument('checkpoint', type=str, default='./ckpt.pth', action='store', help='Select a model to predict a class')
        parser.add_argument('--top_k', type=int, default=5, action='store', choices=[3, 5, 7] ,help='Show the most likely classes')
        parser.add_argument('--category_names', type=str, action='store', help='Show the real name of a flower')
        parser.add_argument('--gpu', action='store_true', default='gpu', help='GPU on and off for training')
        # Parse args
        args = parser.parse_args()
        
        return args
