class DataForTorch:
    def __init__(self, train_dir, valid_dir, test_dir):
        # Object : constructor
        # Input : directory pathes of train_dir, valid_dir, test_dir
        self.__train_dir = train_dir
        self.__valid_dir = valid_dir
        self.__test_dir = test_dir

    def prepare_data(self):
        # Object : public function for setting data for training
        # 3 methods :__transform_data(self), __load_data(self), __load_label(self)
        def __transform_data(self):
            # Object : transform images for training
            import torch
            from torchvision import datasets, transforms, models

            train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

            test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

            # Load the datasets with ImageFolder
            self.__train_datasets = datasets.ImageFolder(self.__train_dir, transform=train_transforms)
            self.__valid_datasets = datasets.ImageFolder(self.__valid_dir, transform=test_transforms)
            self.__test_datasets = datasets.ImageFolder(self.__test_dir, transform=test_transforms)
            self.__class_to_idx = self.__train_datasets.class_to_idx

        def __load_data(self):
            # Object : Load trainloaders for training
            import torch
            from torchvision import datasets, transforms, models
            
            self.__trainloaders = torch.utils.data.DataLoader(self.__train_datasets, batch_size=64, shuffle=True)
            self.__validloaders = torch.utils.data.DataLoader(self.__valid_datasets, batch_size=64)
            self.__testloaders = torch.utils.data.DataLoader(self.__test_datasets, batch_size=64)

        def __load_label(self):
            # Object : Load labels for training
            import json
            with open('cat_to_name.json', 'r') as f:
                self.__cat_to_name = json.load(f)

        __transform_data(self)
        __load_data(self)
        __load_label(self)


        self.__result = [self.__trainloaders, self.__validloaders, self.__testloaders, self.__cat_to_name, self.__train_datasets, self.__class_to_idx]
        return self.__result  
