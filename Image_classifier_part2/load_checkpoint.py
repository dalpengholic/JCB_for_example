class LoadModel:
    def __init__(self, modelname, learning_rate=None):
        # Object : constructor
        # Input : name of model, learning_rate
        self.__modelname = modelname

        if learning_rate is None:
            self.__learning_rate = 0.002
        else :
            self.__learning_rate = learning_rate

    def load_model(self):
        # Object : load a model for training
        import torch
        from torchvision import models
        from torch import nn, optim
        ckpt = torch.load(self.__modelname, map_location=lambda storage, loc: storage)
        self.__model = ckpt['model']

        # Freeze parameters so we don't backprop through them
        # Not freeze parameters of a clssifier that we want to keep training 
        child_counter = 0
        for child in self.__model.children():
            if child_counter == 0:
                for param in child.parameters():
                    param.requires_grad = False
                    child_counter += 1
            
            else:
                for param in child.parameters():
                    param.requires_grad = True

        self.__model.classifier = ckpt['classifier']
        self.__model.load_state_dict(ckpt['state_dict'])
        self.__model.class_to_idx = ckpt['class_to_idx']
        # Set criterion and optimizer
        self.__criterion = nn.NLLLoss()
        self.__optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.__model.parameters()), lr = self.__learning_rate) 
        self.__result = [self.__model, self.__criterion, self.__optimizer, self.__model.class_to_idx]    
        
        return self.__result 