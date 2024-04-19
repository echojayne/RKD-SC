from models.model import Classifier

class NetworkManager:

    # usage:
    # net = NetworkManager(512, 100)
    # classifier = net.create_network([256, 128])
    # print(classifier)

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def create_network(self, layer_sizes):
        layer_sizes = [self.input_size] + layer_sizes + [self.output_size]
        return Classifier(layer_sizes)
    
