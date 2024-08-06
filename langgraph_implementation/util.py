
model_costs = {
    "gpt-4": {
        "input": 0.00003,
        "output": 0.00006
    },
    "gpt-3.5-turbo": {
        "input": 0.0000005,
        "output": 0.0000015
    },
    "gpt-4-turbo": {
        "input": 0.00001,
        "output": 0.00003
    },
    "gpt-4o": {
        "input": 0.00005,
        "cost": 0.000015
    }
}


def sum_tokens(data):
    prompt_tokens, completion_tokens = 0,0
    for toks in data:
        prompt_tokens += toks[0]
        completion_tokens += toks[1]
    return prompt_tokens, completion_tokens


def calculate_cost(input_tokens, output_tokens, model_name):
    if model_name not in model_costs:
        return 0
    cost = model_costs.get(model_name).get("input") * input_tokens + \
           model_costs.get(model_name).get("output") * output_tokens

    return cost
