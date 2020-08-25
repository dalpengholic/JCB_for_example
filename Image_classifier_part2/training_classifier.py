class TrainModel:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models
    from workspace_utils import active_session
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, data_for_model_list, model_to_train_list, gpu=False, epochs=None):
        # Object : constructor
        # Input : prepaired data, a model from new one or saved one, gpu, epochs
        self.__trainloaders = data_for_model_list[0]
        self.__validloaders = data_for_model_list[0]
        self.__model = model_to_train_list[0]
        self.__criterion = model_to_train_list[1]
        self.__optimizer = model_to_train_list[2]
        self.__class_to_idx = data_for_model_list[5]
        self.__device = TrainModel.device
        self.__gpu = gpu
        if epochs is None:
            self.__epochs = 2
        else :
            self.__epochs = epochs
        
    def train_model(self):
        # Object : public function for training a model
        # 3 methods :__model_to_device(self) __train_model(self)
        def __model_to_device(self):
            import torch
            self.__model.to(self.__device)

        def __train_model(self):
            from workspace_utils import active_session
            import torch
            from torchvision import datasets, transforms, models
            with active_session():
                # Do long-running work here
                steps = 0
                running_loss = 0
                print_every = 10
                for epoch in range(self.__epochs):
                    for inputs, labels in self.__trainloaders:
                        steps += 1
                        inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                        self.__optimizer.zero_grad()
                    
                        logps = self.__model.forward(inputs)
                        loss = self.__criterion(logps, labels)
                        loss.backward()
                        self.__optimizer.step()
                    
                        running_loss += loss.item()
                    
                        if steps % print_every == 0:
                            valid_loss = 0
                            accuracy = 0
                            self.__model.eval()
                            with torch.no_grad():
                                for inputs, labels in self.__validloaders:
                                    inputs, labels = inputs.to(self.__device), labels.to(self.__device)
                                    logps = self.__model.forward(inputs)
                                    batch_loss = self.__criterion(logps, labels)
                                    valid_loss += batch_loss.item()
                                
                                    # Calculate accuracy
                                    ps = torch.exp(logps)
                                    top_p, top_class = ps.topk(1, dim=1)
                                    equals = top_class == labels.view(*top_class.shape)
                                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                                
                            print(f"Epoch {self.__epochs+1}/{self.__epochs}.. "
                                f"Train loss: {running_loss/print_every:.3f}.. "
                                f"Valid loss: {valid_loss/len(self.__validloaders):.3f}.. "
                                f"Valid accuracy: {accuracy/len(self.__validloaders):.3f}")
                            running_loss = 0
                 
                        self.__model.train()

        if self.__gpu:
            __model_to_device(self)
            __train_model(self)

        else:
            __train_model(self)
        
        self.__result = [self.__epochs, self.__model, self.__class_to_idx]
        return self.__result

