from memristor.devices import StaticMemristor, DynamicMemristor, DynamicMemristorFreeRange, DynamicMemristorStuck
from memristor.crossbar.model import LineResistanceCrossbar
import torch

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
        - __init__(): Constructor - As if you are ordering a physical chip, specify the specs of the chip
        - register_weights(): Directly specify crossbar memristor weights and calibrate ADCs
        - program_crossbar(): Apply pulses to modify crossbar memristor weights and calibrate ADCs
        - inference(): Compute VMM

    include the convert real value number into binary/whatever format into each function where it's used.
    When initializing the entire engine, specify the format of number / fixed-points and stuff
    """

    def __init__(self, number_int_bits: int, number_frac_bits: int, twos_complement: bool,
                 adc_class,
                 crossbar_class, crossbar_params, memristor_model_class, memristor_params, initial_weight,
                 total_rows: int, total_cols: int):
        """
        Initialize the VMM Engine. This should define the expected behavior of the engine.
        :param number_int_bits: Assume fixed-point numbers. Specify number of integer bits
        :param number_frac_bits: Assume fixed-point numbers. Specify number of fractional bits.
        :param twos_complement: If true, the first bit will represent -(2 ** (number_int_bits - 1))
                                If false, the first bit will represent +(2 ** (number_int_bits - 1))
        :param adc_class: The ADC class to use for the engine. We assume the whole engine uses the same type of ADC
                          But we can have different ADCs for different crossbars. (For calibration purposes)
        :param crossbar_class: The crossbar class to use for the engine. We assume the whole engine uses the same TYPE
                               of crossbar.
        :param crossbar_params: The parameters to pass to the crossbar class.
        :param memristor_model_class: The memristor class to use for the engine. We assume the whole engine uses the
                                      same TYPE of memristor.
        :param memristor_params: The parameters to pass to the memristor class.
        :param initial_weight: The initial weight to set the memristors to when building the crossbar.
        """
        self.number_int_bits = number_int_bits
        self.number_frac_bits = number_frac_bits
        self.twos_complement = twos_complement
        self.adc_class = adc_class
        self.crossbar_class = crossbar_class
        self.crossbar_params = crossbar_params
        self.memristor_model_class = memristor_model_class
        self.memristor_params = memristor_params
        self.initial_weight = initial_weight
        self.total_rows = total_rows
        self.total_cols = total_cols

    def register_weights(self, weights):
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


class BinarySemiPassiveVMMEngine(VMMEngine):
    """
    TODO: I seem to be making this into a dot product engine instead of VMM engine.
          FIX for inference: count the column index, make sure to reset column index every "int_bit+frac_bit" cycles
    """

    def __init__(self, number_int_bits: int, number_frac_bits: int, twos_complement: bool,
                 adc_class,
                 crossbar_class, crossbar_params, memristor_model_class, memristor_params, initial_weight,
                 total_rows: int, total_cols: int,
                 crossbars_row_splits: int = 1, crossbars_col_splits: int = 1):
        """
        Initialize the VMM Engine. This should define the expected behavior of the engine.
        :param number_int_bits: Assume fixed-point numbers. Specify number of integer bits
        :param number_frac_bits: Assume fixed-point numbers. Specify number of fractional bits.
        :param twos_complement: If true, the first bit will represent -(2 ** (number_int_bits - 1))
                                If false, the first bit will represent +(2 ** (number_int_bits - 1))
        :param adc_class: The ADC class to use for the engine. We assume the whole engine uses the same type of ADC
                          But we can have different ADCs for different crossbars. (For calibration purposes)
        :param crossbar_class: The crossbar class to use for the engine. We assume the whole engine uses the same TYPE
                               of crossbar.
        :param crossbar_params: The parameters to pass to the crossbar class.
        :param memristor_model_class: The memristor class to use for the engine. We assume the whole engine uses the
                                      same TYPE of memristor.
        :param memristor_params: The parameters to pass to the memristor class.
        :param initial_weight: The initial weight to set the memristors to when building the crossbar.
        :param total_rows: Assume the engine is a singular crossbar. The total number of rows in the crossbar.
        :param total_cols: Assume the engine is a singular crossbar. The total number of columns in the crossbar.
        :param crossbars_row_splits: Split the crossbar into multiple crossbars. The number of crossbars rows.
        :param crossbars_col_splits: Split the crossbar into multiple crossbars. The number of crossbars columns.
        """
        # TODO: do we need to set float64?
        # TODO: can let user input "number of numbers per row of of total crossbar
        super().__init__(number_int_bits, number_frac_bits, twos_complement,
                         adc_class,
                         crossbar_class, crossbar_params, memristor_model_class, memristor_params, initial_weight,
                         total_rows, total_cols)
        self.crossbars_row_splits = crossbars_row_splits
        self.crossbars_col_splits = crossbars_col_splits

        # Determine the m and n for each of the (crossbars_row_splits * crossbars_col_splits) crossbars
        self.average_rows_per_row_split = self.total_rows // self.crossbars_row_splits
        self.average_cols_per_col_split = self.total_cols // self.crossbars_col_splits
        self.number_of_row_splits_with_extra_rows = self.total_rows % self.crossbars_row_splits
        self.number_of_col_splits_with_extra_cols = self.total_cols % self.crossbars_col_splits
        self.crossbars_n_m = []
        # We put larger crossbars in the top left corner
        for i in range(crossbars_row_splits):
            self.crossbars_n_m.append([])
            for j in range(crossbars_col_splits):
                self.crossbars_n_m[i].append(
                    (self.average_rows_per_row_split + (1 if i < self.number_of_row_splits_with_extra_rows else 0),
                     self.average_cols_per_col_split + (1 if j < self.number_of_col_splits_with_extra_cols else 0)))

        # Initialize the crossbars
        self.crossbars = []
        for i in range(crossbars_row_splits):
            self.crossbars.append([])
            for j in range(crossbars_col_splits):
                ideal_w = torch.ones(self.crossbars_n_m[i][j]) * self.initial_weight
                self.crossbars[i].append(
                    self.crossbar_class(
                        self.memristor_model_class, self.memristor_params, ideal_w, self.crossbar_params
                    )
                )

    def register_weights(self, weights):
        """
        Initialize the crossbar with the given weights. We
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
