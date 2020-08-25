class SanityChecker:

    def __init__(self, guess_list, model_to_train_list, f_name=None):
        # object : constructor
        # input : result of prediction, model, json file for mapping from classes to names
        self.__top_p = guess_list[0]
        self.__top_class = guess_list[1]
        self.__class_to_idx = model_to_train_list[3]
        if f_name is None:
            self.__f_name = 5
        else :
            self.__f_name = f_name
        self.__key_list = []
        self.__flowername_list = []

    def index_to_flowername(self):
        # object : map classes to the names of flowers
        # 3 methods :__load_label(self), __get_key(self),__get_flowername(self)
        def __load_label(self):
            import json
            with open(self.__f_name, 'r') as f:
                self.__cat_to_name = json.load(f)

        def __get_key(self):
            self.__class_to_idx = dict((v,k) for k,v in self.__class_to_idx.items())
            for i in self.__top_class:
                j = self.__class_to_idx.get(i)
                self.__key_list.append(j)

                
        def __get_flowername(self):
            for i in self.__key_list:
                self.__name = self.__cat_to_name.get(i)
                self.__flowername_list.append(self.__name)
                
      
        __load_label(self)
        __get_key(self)
        __get_flowername(self)

        self.__result = [self.__flowername_list]
        return self.__result

