class ImageClassifier:
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, processed_image, model_to_train_list, topk=None, gpu=False):
        # Object : constructor
        # Input : processed image, model, the number of class to print, gpu
        self.__processed_image = processed_image
        self.__model = model_to_train_list[0]
        if topk is None:
            self.__topk = 5
        else :
            self.__topk = topk
        self.__device = ImageClassifier.device
        self.__gpu = gpu
        
    def predict(self):
        # Object : predict probabilities and classes of an input
        # 3 methods :__model_to_device(self) __predict_category(self)
        def __model_to_device(self):
            import torch
            # Model to GPU
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.__model.to(self.__device)
            self.__model.eval()

        def __predict_category(self):
            import torch
            import numpy as np
            input_image = torch.from_numpy(self.__processed_image)
            input_image.unsqueeze_(0)

            with torch.no_grad():
                input_image = input_image.to(device=self.__device, dtype=torch.float)
                logps = self.__model.forward(input_image)
                ps = torch.exp(logps)
                self.__top_p, self.__top_class = ps.topk(self.__topk, dim=1)
            
            self.__top_p = np.array(self.__top_p).flatten()
            self.__top_class = np.array(self.__top_class).flatten()


        if self.__gpu:
            __model_to_device(self)
            __predict_category(self)

        else:
            __predict_category(self)
        
        self.__result = [self.__top_p, self.__top_class]    
        return self.__result
  
   