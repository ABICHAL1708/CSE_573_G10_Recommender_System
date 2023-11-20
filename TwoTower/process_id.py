from test import *

def process_id(input_value, selected_option):
    result = f"You entered: {input_value}"
    return get_recommendation(int(input_value), 10)