"""
    [INST]You are a text classification bot. 
    Your task is to assess intent and categorize the input 
    text after <<<>>> into one of the following predefined categories:

    2: Presence - Present / default
    1: Presence - Hypothetical / not applicable
    0: Not present

    Explanation of labels:
    Label 2 (present / default) is the class where the context strongly indicates that the given medical entity is present. The text will not explicitly contain mention that it is present, you have to infer it.  
    Label 1 (hypothetical) is the class where the context clearly indicates that the given medical entity is not applicable / hypothetical.
    Label 0 (not present) is when the context clearly indicates that the given medical entity is not present
    
    You will only respond with the predefined category. 
    Do not provide explanations or notes. 
    
    Format of input: 'tokens', 'medical entity'. Classify the text for the medical entity in the tokens.
    <<<
    Inquiry: {input}
    >>>
    Category: [/INST]
    """
