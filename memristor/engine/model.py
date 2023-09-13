

class VMMEngine:
    """
    Abstract class for VMM Engine

    Functions:
        - __init__(): Constructor
        - build_crossbar(): Initialize the crossbars
        - register_weights(): Directly specify crossbar memristor weights
        - inference(): Compute VMM
        - program_crossbar(): Apply pulses to modify crossbar memristor weights
    """
    def __init__(self):
        pass

    def build_crossbar(self, crossbar):
        pass

    def register_weights(self, crossbar, weights):
        pass

    def inference(self, crossbar, input_vector):
        pass

    def program_crossbar(self, crossbar, input_vector, output_vector):
        pass


class BinarySemiPassiveVMMEngine:

    def __init__(self):
        pass

    def build_crossbar(self, crossbar):

        pass

    def register_weights(self, crossbar, weights):
        pass

    def inference(self, crossbar, input_vector):
        pass

    def program_crossbar(self, crossbar, input_vector, output_vector):
        pass
