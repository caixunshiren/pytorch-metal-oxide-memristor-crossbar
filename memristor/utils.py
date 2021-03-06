from pathlib import Path
import pandas as pd


class DynamicParams:
    COLUMNS = ["c0", "c1", "c2", "c3", "c4", "d0", "d1", "d2", "d3", "d4"]
    def __init__(self):
        self.set = None
        self.reset = None
        self.load_dyanamic_params()

    def load_dyanamic_params(self):
        path = Path(__file__).parent
        df = pd.read_csv(path/"dynamic_params.txt", sep=' ', header=None)
        self.set = df.iloc[:, range(10)]
        self.reset = df.iloc[:, [10+i for i in range(10)]]
        self.set.set_axis(self.COLUMNS, axis=1, inplace=True)
        self.reset.set_axis(self.COLUMNS, axis=1, inplace=True)

    def get_params(self, g_0):
        """
        get appropriate parameters based on the conductance
        :param g_0: conductance of the device
        :return: appropriate conductance dictionary
        """
        pass

dp = DynamicParams()
print(dp.set)
print(dp.reset)
