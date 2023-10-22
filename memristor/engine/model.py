from memristor.devices import StaticMemristor, DynamicMemristor, DynamicMemristorFreeRange, DynamicMemristorStuck
from memristor.crossbar.model import LineResistanceCrossbar
import torch
import numpy as np

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

    def inference(self, input_vector):
        """
        input ONE vector, output ONE vector.
        - do whatever shift/add thing we need ADC and stuff

        param:
        - ADC param?
        """
        pass

    def program_crossbar(self, weights: torch.tensor):
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
        # TODO: can let user input "number of binary numbers per row of crossbar" and auto-populates the "total_cols"
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
        self.crossbars_m_n = []
        # We put larger crossbars in the top left corner
        for i in range(self.crossbars_row_splits):
            self.crossbars_m_n.append([])
            for j in range(self.crossbars_col_splits):
                self.crossbars_m_n[i].append(
                    (self.average_rows_per_row_split + (1 if i < self.number_of_row_splits_with_extra_rows else 0),
                     self.average_cols_per_col_split + (1 if j < self.number_of_col_splits_with_extra_cols else 0)))

        # Initialize the crossbars
        self.crossbars = []
        for i in range(self.crossbars_row_splits):
            self.crossbars.append([])
            for j in range(self.crossbars_col_splits):
                ideal_w = torch.ones((self.crossbars_m_n[i][j][1], self.crossbars_m_n[i][j][0])) * self.initial_weight
                self.crossbars[i].append(
                    self.crossbar_class(
                        self.memristor_model_class, self.memristor_params, ideal_w, self.crossbar_params
                    )
                )

        # Initialize the ADCs. One ADC per crossbar
        self.adcs = []
        for i in range(self.crossbars_row_splits):
            self.adcs.append([])
            for j in range(self.crossbars_col_splits):
                self.adcs[i].append(self.adc_class())
                # TODO: pass the adc params, such as num_bits, precisions, etc.

        # Initialize the variable tracking the number of values stored in the crossbar
        # These are to be updated every time we modify the crossbar
        self.number_of_row_values = 0  # How many rows are in the weight matrix
        self.number_of_col_values = 0  # How many columns are in the weight matrix

    def convert_number_to_binary(self, number):
        """
        Convert a number to a binary representation.
        Check if the number is too large to be represented with the given number of bits.
        Check if number is negative, if so, check if we use two's complement.

        :param number: The number to convert.
        :return: The binary representation of the number, using the number of integer and fractional bits specified
                 in the constructor. in list form: [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
        """
        int_representation = round(number * (2 ** self.number_frac_bits))

        # check if the number is too large to be represented
        if self.twos_complement:
            assert int_representation < 2 ** (self.number_int_bits + self.number_frac_bits - 1), \
                "Number too large to be represented with the given number of bits"
            assert int_representation >= -2 ** (self.number_int_bits + self.number_frac_bits - 1), \
                "Number too negative to be represented with the given number of bits"
        else:
            assert int_representation < 2 ** (self.number_int_bits + self.number_frac_bits), \
                "Number too large to be represented with the given number of bits"
            assert int_representation >= 0, \
                "Negative numbers not supported without two's complement"

        bit_representation = [0] * (self.number_int_bits + self.number_frac_bits)

        # Set MSB if negative
        if int_representation < 0:
            bit_representation[0] = 1
            int_representation += 2 ** (self.number_int_bits + self.number_frac_bits - 1)

        # Start from the LSB, skip MSB if twos_complement
        for i in range(self.number_int_bits + self.number_frac_bits):
            if i == self.number_int_bits + self.number_frac_bits - 1 and self.twos_complement:
                break
            bit_representation[self.number_int_bits + self.number_frac_bits - 1 - i] = int_representation % 2
            int_representation //= 2

        return bit_representation

    def convert_2d_list_to_binary(self, list_2d: torch.tensor):
        """
        Convert a 2D list of numbers to a 2D list of binary representation.
        :param list_2d: The 2D list of numbers to convert.
        :return: The 2D list of binary representation of the numbers, using the number of integer and fractional bits
                 specified in the constructor.
                 Shape: [list_2d.shape[0], list_2d.shape[1] * (number_int_bits + number_frac_bits)]
        """
        binary_tensor = torch.zeros(list_2d.shape[0], list_2d.shape[1], self.number_int_bits + self.number_frac_bits)
        for i in range(list_2d.shape[0]):
            row_binary = []
            for j in range(list_2d.shape[1]):
                row_binary += self.convert_number_to_binary(list_2d[i, j])
            binary_tensor[i] = torch.tensor(row_binary)
        return binary_tensor

    def register_weights(self, weights: torch.tensor):
        """
        TODO: specify the conductance range of the crossbar. Specifically, the maximum and minimum conductance for 1, 0.
        Initialize the crossbar with the given weights. In binary form, 1 is LRS and 0 is HRS.
        weights is "A" in Ax = b, where x is the input vector and b is the output vector.
        First column of weights is the first row of the crossbar, second column of weights is the second row...
        So we first transpose the weights matrix.

        We convert weights into [weights.m, weights.n * (number_int_bits + number_frac_bits)] binary matrix.
        When inference, pay attention to not shift-add different numbers together.

        First check if the weights are in the correct shape.
            Only raise error if weight is too large. If weight is too small, pad with zeros.

        Then register the crossbars one by one with the weights.

        Calibrate the ADCs.
        """
        # Transpose the weights matrix.
        weights = weights.transpose(0, 1)  # After this, first row of weights is the first row of the crossbar, etc.

        weights_m, weights_n = weights.shape
        assert weights_m <= self.total_rows, "Weight matrix too many rows for crossbar"
        assert weights_n * (self.number_int_bits + self.number_frac_bits) <= self.total_cols, \
            "Weight matrix too many columns for crossbar"

        # convert weights to binary
        weights_binary = self.convert_2d_list_to_binary(weights)

        # Update the number of values stored in the crossbar - for inference
        self.number_of_row_values = weights_m  # How many rows are in the weight matrix
        self.number_of_col_values = weights_n  # How many columns are in the weight matrix

        # pad with zeros if necessary
        if weights_binary.shape[0] < self.total_rows:
            weights_binary = torch.cat(
                (weights_binary, torch.zeros(self.total_rows - weights_m, weights_n * (self.number_int_bits + self.number_frac_bits))),
                dim=0
            )
        if weights_binary.shape[1] < self.total_cols:
            weights_binary = torch.cat(
                (weights_binary, torch.zeros(self.total_rows, self.total_cols - weights_n * (self.number_int_bits + self.number_frac_bits))),
                dim=1
            )

        # register weights. Iterate crossbar by crossbar.
        # crossbar class requires weights to be in [n, m] shape. We transpose the weights here.
        # TODO: we transposed twice here, can we do it in one go?
        weights_binary = weights_binary.transpose(0, 1)  # this converts [m,n] to [n,m]
        row_start_index = 0
        col_start_index = 0
        for i in range(self.crossbars_row_splits):
            for j in range(self.crossbars_col_splits):
                # TODO: currently we register weights of 1 or 0... which is incorrect.
                #   Must convert weights to the appropriate g_0 values (like 3.16e-6 S)
                self.crossbars[i][j].register_weights(weights_binary[row_start_index:row_start_index + self.crossbar_rows,
                                                                     col_start_index:col_start_index + self.crossbar_cols])
                # TODO: calibrate ADCs here (need to give crossbar, binary weights, and possible inputs?)
                #   self.adcs[i][j].calibrate(self.crossbars[i][j], weights_binary[row_start_index:row_start_index + self.crossbar_rows,
                #                                                                      col_start_index:col_start_index + self.crossbar_cols])
                col_start_index += self.crossbars_m_n[i][j][1]
            row_start_index += self.crossbars_m_n[i][0][0]
            col_start_index = 0

    def inference(self, input_tensor: torch.tensor):
        """
        TODO: specify the inference voltage, specifically a voltage for 1 and a voltage for 0. (I would assume V_0 = 0V)
        Convert input vector to binary, from (m,) to (m, number_int_bits + number_frac_bits).
        Check that "m" matches self.number_of_row_values. if too small, pad with zeros. too large, raise error.

        Iterate through (number_int_bits + number_frac_bits) times for input_vector. Each time input:
            input_vector_binary[:, i] is the input vector for the crossbar.
            (start from MSB, so the 0th input can be adjusted in shift-add by 2's complement)

            For each input, iterate through crossbars. Row split by row split.
                For each row split, record the ADC output of each bit-line into one list.
                Then shift-add each bit-line result. (bit-line-result ^ 2^bit-line-number-counting-from-right)
                (for left-most bit-line result, always check if two's complement. If so, subtract 2^(number_total_bits-1))
                Make sure you reset bit-line-count to 0 after each number is checked.
                    if there are 32 bit-lines, but your numbers are 16-bit binary,
                    reset bit-line-count to 0 after every 16 bit-lines. (cuz it's a new number)

            Sum up result from each row split. (because the operation across rows are just addition in crossbars)
                Make sure to distinguish between different numbers' binary representation across bit-lines.
                only sum up the bit-lines that belong to the same number.

            Shift-add each input-bit's result. But shift-subtract the MSB input-bit's result if it is 2's complement.

        divide final number by 2^(2 * number_frac_bits) to get the final result.
        """
        pass

    def program_crossbar(self, weights: torch.tensor):
        """
        TODO: specify program voltage, pulse width, num_pulses etc.
            Specify programming cell by cell (On^2) or row by row (On).
        Program the crossbar with the given weights. In binary form, 1 is LRS and 0 is HRS.

        Transpose the weights matrix.

        Convert weights into [weights.m, weights.n * (number_int_bits + number_frac_bits)] binary matrix.

        Iterate through crossbars splits, program each crossbar one by one.
            Use a positive pulse + negative pulse to program each crossbar.

        Calibrate ADCs after programming.
        """
        # TODO: calibrate ADCs here (need to give crossbar, binary weights, and possible inputs?)
        #   self.adcs[i][j].calibrate(self.crossbars[i][j], weights_binary[row_start_index:row_start_index + self.crossbar_rows,
        #                                                                      col_start_index:col_start_index + self.crossbar_cols])
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


class NaiveLSH:
    # so the idea is that the crossbar contains the randomly
    # generated weights which are used for the random projection
    # stuff for LSH


    class LSHTable:
        def __init__(self, hash_size, input_dimensions, crossbar, a, b, c, d):
            self.hash_size = hash_size # how many bits
            self.input_dimensions = input_dimensions
            self.hash_table = {}
            self.crossbar = crossbar
            self.a = a
            self.b = b
            self.c = c
            self.d = d
        
        def generate_hash(self, input_vector):
            input_vector_with_beta = np.vstack((input_vector, np.array([[1]])))



            ###bools = (self.crossbar.ideal_vmm(torch.from_numpy(input_vector_with_beta)).numpy() > 0).astype('int').squeeze()
            
            
            result = self.crossbar.ideal_vmm(torch.from_numpy(input_vector_with_beta)).numpy()
            #print (result.shape) # (5, 1) so don't need to reshape      5 is hash size
            b_broadcast = np.full(self.crossbar.ideal_w.numpy().shape, self.b)
            d_broadcast = np.full(input_vector_with_beta.shape, self.d)
            converted = (result - self.c*(b_broadcast@((input_vector_with_beta-self.d)/self.c)) - self.a*(((self.crossbar.ideal_w.numpy()-self.b)/self.a)@d_broadcast) - b_broadcast@d_broadcast) / self.a*self.c
            #                                                                  ^ should automatically broadcast,                           ^ this too
            bools = (converted > 0).astype('int').squeeze()
            
            
            return ''.join(bools.astype('str'))

            
            
            
            '''


            # bools = (self.crossbar.naive_memristive_vmm(input_vector_with_beta).numpy() > 0).astype('int')
            # bools2 = (self.crossbar.naive_memristive_vmm(input_vector_with_beta).numpy() > 0).astype('int')
            # print (type(bools2))
            # print (bools2)
            # print (bools2.astype('str'))
            #bools = (self.crossbar.ideal_vmm(torch.from_numpy(input_vector_with_beta)).numpy() > 0).astype('int').squeeze()
            # print (type(bools))
            # print (bools)
            # print (bools.astype('str'))
            # print (bools.squeeze().astype('str'))
            result = self.crossbar.naive_memristive_vmm(input_vector_with_beta).numpy().reshape(-1, 1)
            # print (type(result))
            # print (type(input_vector))
            # print (type(self.crossbar.ideal_w))
            # converted = (result - self.b*self.c*((input_vector-self.d)/self.c) + self.a*self.d*((self.crossbar.ideal_w.numpy()-self.b)/self.a) + self.b*self.d) / self.a*self.c
            # for i in [self.a, self.b, self.c, self.d]:
            #     print (i) # scalars
            # print (input_vector_with_beta.shape) # (6,1)
            # print (self.crossbar.ideal_w.numpy().shape) # (3,6)
            # print (result.shape) # (3,)   should be 3x1?
            # converted = (result - self.b*self.c*((input_vector_with_beta-self.d)/self.c) + self.a*self.d*((self.crossbar.ideal_w.numpy()-self.b)/self.a) + self.b*self.d) / self.a*self.c
            # b_broadcast = np.full((3, 6), self.b)
            # d_broadcast = np.full((6, 1), self.d)
            b_broadcast = np.full(self.crossbar.ideal_w.numpy().shape, self.b)
            d_broadcast = np.full(input_vector_with_beta.shape, self.d)
            converted = (result - self.c*(b_broadcast@((input_vector_with_beta-self.d)/self.c)) - self.a*(((self.crossbar.ideal_w.numpy()-self.b)/self.a)@d_broadcast) - b_broadcast@d_broadcast) / self.a*self.c
            #                                                                  ^ should automatically broadcast,                           ^ this too
            # print (converted.shape)
            # print (result.shape) # (3,)
            # print ((self.c*(b_broadcast@((input_vector_with_beta-self.d)/self.c))).shape) # (3,1)
            # print (result)
            # print (result.reshape(-1, 1))
            # print (result.reshape(-1, 1).shape)
            # bools = (converted > 0).astype('int')
            bools = (converted > 0).astype('int').squeeze()
            # print (bools)
            return ''.join(bools.astype('str'))
            '''
        
        def __setitem__(self, input_vec, label):
            hash_value = self.generate_hash(input_vec)
            self.hash_table[hash_value] = self.hash_table.get(hash_value, []) + [label]

        def __getitem__(self, input_vec):
            hash_value = self.generate_hash(input_vec)
            return self.hash_table.get(hash_value, [])  


    def __init__(self,
                 hash_size,
                 crossbar_class,
                 crossbar_params,
                 memristor_model_class,
                 memristor_params,
                 m,
                 r,
                 ):
        self.hash_size = hash_size
        # initialize the crossbar with random weights:
        # ideal_w = np.random.uniform(10e-6, 300e-6, (hash_size, m)) # does the precision matter like double or float or whatever
        ideal_w = np.random.randn(hash_size, m)
        # random values needed by the lsh algorithm are np.random.randn values, we need to map them to 10e-6 to 300e-6 conductance values, to deal with negative values use differential mapping
        #beta = np.random.uniform(0, r, (hash_size, 1))
        # for simplicity, rn gonna get beta from standard normal distribution too:
        beta = np.random.randn(hash_size, 1)
        ideal_w_with_beta = torch.from_numpy(np.concatenate((ideal_w, beta), axis=1))
        # ^ so this is a tensor with real number weights, including negative values      (with beta also drawn from std normal, all the values are drawn from std normal rn)
        # we need to map this to the conductance range of 10e-6 to 300e-6:
        #print (ideal_w_with_beta.shape)
        #raise ValueError
        min_w = ideal_w_with_beta.min().item()
        max_w = ideal_w_with_beta.max().item()
        min_g = 10e-6
        max_g = 300e-6
        a = (max_g - min_g) / (max_w - min_w)
        b = min_g - a * min_w
        ideal_g_with_beta = a * ideal_w_with_beta + b
        self.a = a
        self.b = b
        self.c = None
        self.d = None
        self.crossbar = crossbar_class(
            memristor_model_class, memristor_params, ideal_g_with_beta, crossbar_params ####################################################### g not w
        )
    
    def register_weights(self, weights):
        # for this VMM, weights are always randomly initialized 
        # in the init and should not be changed (since they need
        # to be random for LSH to work)
        raise NotImplementedError
    
    def inference(self, input_vector):
        # here input_vector is the queries or keys, and the goal
        # is to separate them into buckets based on their hashes
        # so that attention will be done at each bucket; so the 
        # output of this function is the computed hash of the 
        # input_vector based on the rows/columns of the weights
        # of the crossbar (not sure if it should be rows or columns)
        min = np.min(input_vector)
        max = np.max(input_vector)
        v_min = -0.4
        v_max = 0.4
        c = (v_max - v_min) / (max - min)
        d = v_min - c * min
        input_v = c * input_vector + d
        self.c = c
        self.d = d
        lsh_table = NaiveLSH.LSHTable(
            hash_size=self.hash_size,
            input_dimensions=input_v.shape[0], # vector
            # ^ this needs to be the same as m in the init
            crossbar=self.crossbar,
            a=self.a,
            b=self.b,
            c=self.c,
            d=self.d,
        )
        return lsh_table.generate_hash(input_v) # vector
    
    def program_crossbar(self, weights: torch.tensor):
        raise NotImplementedError
    
    def hash_fn(self, queries, keys):
        # assuming queries and keys are lists containing vectors
        # of the right shape based on the crossbar size
        buckets = {}
        for query in queries:
            hash = self.inference(query)
            if hash not in buckets.keys():
                buckets[hash] = {"queries": [query], "keys": []}
            else:
                buckets[hash]["queries"].append(query)
        for key in keys:
            hash = self.inference(key)
            if hash not in buckets.keys():
                buckets[hash] = {"queries": [], "keys": [key]}
            else:
                buckets[hash]["keys"].append(key)
        return buckets