# Add shared components here, such as ADC

class ADC:
    def __init__(self):
        pass

    def calibrate(self, crossbar, crossbar_expected_weights):
        # pass a single crossbar and what each weight on crossbar should be.
        # Input to crossbar all possible inputs, and average the output.
        # This dictionary of "lookup" should be stored in the ADC object.
        pass

    def decode(self, crossbar_output_current):
        # pass the bitlines current, and return the decoded value. Output should be a list of integers.
        pass
