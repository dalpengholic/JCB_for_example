class BuildModel:
    # Set defaults units for each model
    initial_units = {"vgg16":25088, "vgg13":25088, "densenet121":1024}

    def __init__(self, architecture=None, learning_rate=None, hidden_units=None):
        # Object : constructor
        # Input : architecture of model, learning_Rate, hidden_units
        if architecture is None:
            self.__architecture = 'vgg16' 
        else:
            self.__architecture = architecture

        if learning_rate is None:
            self.__learning_rate = 0.002
        else :
            self.__learning_rate = learning_rate

        if hidden_units is None:
            self.__hidden_units = 4096
        else : 
            self.__hidden_units = hidden_units


    def build_model(self):
        # load model from torchvision
        from torchvision import datasets, transforms, models
        from torch import nn, optim
        if self.__architecture == 'vgg16':
            model = models.vgg16(pretrained=True)

        elif self.__architecture == 'vgg13':
            model = models.vgg13(pretrained=True)

        elif self.__architecture == 'densenet121':
            model = models.densenet121(pretrained=True)



        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        # Set a classifier
        classifier = nn.Sequential(nn.Linear(BuildModel.initial_units[self.__architecture],self.__hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.__hidden_units,1024),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(1024,102),
                                    nn.LogSoftmax(dim=1))
                                    
        model.classifier = classifier
        self.__model = model
        # Set a criterion and an optimizer
        self.__criterion = nn.NLLLoss()
        self.__optimizer = optim.Adam(model.classifier.parameters(), lr = self.__learning_rate)
        
        self.__result = [self.__model, self.__criterion, self.__optimizer]
        return self.__result  
