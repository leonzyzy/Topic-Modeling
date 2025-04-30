 """
    Transformer-based sequence model with a skip connection from the last valid timestep.

    Input Shape:
        x: Tensor of shape (B, S, E)
           - B: batch size
           - S: sequence length
           - E: input feature dimension
           - Padding is assumed to be zero-valued rows.

    Architecture:
        - Projects input features E → d_model
        - Applies Transformer encoder over (B, S, d_model)
        - Transposes to (B, d_model, S)
        - Applies a Linear layer to reduce sequence dim S → 1: shape becomes (B, d_model, 1)
        - Squeezes to (B, d_model)
        - Extracts the last non-zero (non-padding) timestep from the original input: shape (B, E)
        - Concatenates to form (B, d_model + E)
        - Applies final Linear layer → (B, 1)

    Output:
        - Tensor of shape (B, 1): one prediction per sequence
    """
