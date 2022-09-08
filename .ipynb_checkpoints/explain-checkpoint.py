class MultiInputWrapper:
    def __init__(self, fixed_inputs, variable_input_index):
        """Class to create a function for DIANNA that
           handles having multiple inputs, of which only one is perturbed
           by the XAI method.
           The initialized class can be given as preprocessing_function to DIANNA

           Args:
                fixed_inputs (list): The inputs that will not be changed by the XAI method
                variable_input_index (int): The index in the fixed_inputs list where the
                    input that will be changed by the XAI method should be inserted

           Example usage:
               preprocess_function = MultiInputWrapper(fixed_inputs, variable_input_index)
               heatmaps = dianna.explain_image(model_runner, variable_input, labels=(0, 1),
                                               method="RISE",
                                               preprocess_function=preprocess_funcion)

        """
        self.fixed_inputs = fixed_inputs
        self.variable_input_index = variable_input_index
        # sanity check on variable_input_index: it must correspond to
        # a location in the fixed_inputs list
        assert (variable_input_index >= 0) and (variable_input_index <= len(fixed_inputs)), \
            f"The index of the variable input should be between 0 and {len(fixed_inputs)}"

        def __call__(self, variable_input):
            """
            Take the input from this function and combine it with the fixed inputs
            to create a complete list of inputs.

            Args:
                variable_input (numpy array compatible): The perturbed input, including
                    a batch axis as first axis
            """
            output = []
            for item in variable_input:
                output_item = self.fixed_inputs.copy()
                output_item.insert(self.variable_input_index, item)
                output.append(output_item)
            return output
