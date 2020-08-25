class SaveModel:
    def __init__(self, trained_model_list, savepath=None):
        # Object : constructor
        # Input : model info, path for saving
        self.__epochs = trained_model_list[0]
        self.__model = trained_model_list[1]
        self.__class_to_idx = trained_model_list[2]
        if savepath is None:
            self.__savepath = './checkpoint.pth'
        else:
            self.__savepath = savepath


    def save_classifier(self):
        # Object : save a model
        import torch
        from torchvision import datasets, transforms, models
        
        checkpoint = {'output_size': 102,
                'epochs': self.__epochs,
                'batch_size': 64,
                'model':  self.__model,
                'classifier': self.__model.classifier,
                'state_dict': self.__model.state_dict(),
                'class_to_idx': self.__class_to_idx
                }
                
        torch.save(checkpoint, self.__savepath)
        print("model is saved")