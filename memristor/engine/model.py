# Overall objective is:
"""
User do:
    engine = VMMEngine()
    engine.build_crossbar(specs)
    weights = [[1, 10, 24], [2, 2, 2], [3, 4, 5]]
    engine.register_weights(weights) OR engine.program_crossbar(weights)
    input_vector = [1, 2, 3]
    output_vector = engine.inference(input_vector)
    print(output_vector)
"""

class VMMEngine:
    """
    Abstract class for VMM Engine

    Functions:
        - __init__(): Constructor
        - build_crossbar(): Initialize the crossbars
        - register_weights(): Directly specify crossbar memristor weights
        - inference(): Compute VMM
        - program_crossbar(): Apply pulses to modify crossbar memristor weights

    include the convert real value number into binary/whatever format into each function where it's used.
    When initializing the entire engine, specify the format of number / fixed-points and stuff
    """
    def __init__(self, ):
        """
        number of float points, int points, 2's compliment....
        """
        # self.crossbars = []
        # pass in a ADC class.
        pass

    def build_crossbar(self, crossbar):
        """
        Build ALL the crossbars involved in this engine
        Return nothing? the built crossbar is now an attribute of the engine
        Expecting:
        - memristor model
        - default ideal_w value
        - crossbar_params
        - memristor_params
        - engine_params (e.g. number of crossbars, rows, cols)
        - can add kargs...?
        """
        pass

    def register_weights(self, crossbar, weights):
        """
        We assume the crossbar is already built?
        so here we only input a weight matrix. (corresponding to shape of crossbar we have, do a check and raise error or something)
        - if weights are smaller, we can just write a corner of the crossbar, but return something to indicate that it didn't program whole thing?
        - calibrate the ADC after this
        """
        pass

    def inference(self, crossbar, input_vector):
        """
        input ONE vector, output ONE vector.
        - do whatever shift/add thing we need ADC and stuff

        param:
        - ADC param?
        """
        pass

    def program_crossbar(self, crossbar, input_vector, output_vector):
        """

        calibrate ADC after programming. Should also specify the WAY OF PROGRAMMING
        - width of pulse, voltage of pulses, number of pulses, program row by row or cell by cell, etc.
        """
        pass


class BinarySemiPassiveVMMEngine:

    def __init__(self, int_bits, frac_bits, adc_class):
        pass

    def build_crossbar(self, crossbar):

        pass

    def register_weights(self, crossbar, weights):
        pass

    def inference(self, crossbar, input_vector):
        pass

    def program_crossbar(self, crossbar, input_vector, output_vector):
        pass


class BinaryFullyPassiveVMMEngine:

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