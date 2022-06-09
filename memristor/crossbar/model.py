class LineResistanceCrossbar:
    def __init__(self, memristor_model, memristor_params, ideal_w):
        """
        :param memristor_model: memristor model class
        :param memristor_params: dictionary of the model param
        :param ideal_w: numpy/torch matrix of ideal weights to be programed
        """
        self.memristor_model = memristor_model
        self.memristor_params = memristor_params
        memristors = [[initialize_memristor(memristor_model, memristor_params, ideal_w[i, j]) for j in ideal_w.shape[1]]
                      for i in ideal_w.shape[0]]

def initialize_memristor(memristor_model, memristor_params, g_0):
    pass