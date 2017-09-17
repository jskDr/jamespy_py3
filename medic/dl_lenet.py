from medic import dl


class CNN(dl.CNN_LENET):
    def __init__(model, input_shape, nb_classes):
        super().__init__(nb_classes, input_shape)
