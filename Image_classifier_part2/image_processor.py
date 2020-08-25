class ImageProcessor:
    def __init__(self, image_path):
        # Object : constructor
        # Input : path of an image
        self.__image_path = image_path

    def process_image(self):
        # Object : Scales, crops, and normalizes a PIL image for a PyTorch model
        # Return : an Numpy array
        from PIL import Image
        import numpy as np
        
        self.__image = Image.open(self.__image_path)
        
        # Make a thumbnail (the shortest side is 256 pixels, keeping the aspect ratio)
        width = self.__image.size[0]
        height = self.__image.size[1]
        
        if width > height :
            new_width = width * 256/height
            new_height = height * 256/height
            self.__image.thumbnail((new_width, new_height))
        
        elif width < height :
            new_width = width * 256/width
            new_height = height * 256/width
            self.__image.thumbnail((new_width, new_height))
            
        else:
            self.__image.thumbnail((256, 256))
        
        width = self.__image.size[0]
        height = self.__image.size[1]
        
        # Crop the image
        left = (width - 224)/2
        top = (height - 224)/2
        right = (width + 224)/2
        bottom = (height + 224)/2
        im_cropped = self.__image.crop((left, top, right, bottom))
        
        # Map the integers of color to the floats from 0 to 1
        np_im_cropped = np.array(im_cropped)
        np_im_cropped_float = np_im_cropped / 255
        
        # Normalize the image data
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_im_normalized =(np_im_cropped_float - mean)/std
        
        # Reorder the dimension
        self.__np_image_transposed = np.transpose(np_im_normalized, (2, 0, 1))
        self.__result = self.__np_image_transposed
        
        return self.__result

