# Costs are USD per token (multiply by token count).
# Rates reflect approximate OpenAI public pricing as of Jul 2026.
model_costs = {
    "gpt-4": {
        "input": 0.00003,  # $30 / 1M
        "output": 0.00006,  # $60 / 1M
    },
    "gpt-4-turbo": {
        "input": 0.00001,  # $10 / 1M
        "output": 0.00003,  # $30 / 1M
    },
    "gpt-4o": {
        "input": 0.0000025,  # $2.50 / 1M
        "output": 0.00001,  # $10 / 1M
    },
    "gpt-4o-mini": {
        "input": 0.00000015,  # $0.15 / 1M
        "output": 0.0000006,  # $0.60 / 1M
    },
    "gpt-5.1": {
        "input": 0.00000125,  # $1.25 / 1M
        "output": 0.00001,  # $10 / 1M
    },
    "gpt-5.2": {
        "input": 0.00000175,  # $1.75 / 1M
        "output": 0.000014,  # $14 / 1M
    },
    "gpt-5-mini": {
        "input": 0.00000025,  # $0.25 / 1M
        "output": 0.000002,  # $2 / 1M
    },
    "gpt-3.5-turbo": {
        "input": 0.0000005,
        "output": 0.0000015,
    },
}


def sum_tokens(data):
    prompt_tokens, completion_tokens = 0, 0
    for toks in data:
        prompt_tokens += toks[0]
        completion_tokens += toks[1]
    return prompt_tokens, completion_tokens


def calculate_cost(input_tokens, output_tokens, model_name):
    if model_name not in model_costs:
        return 0
    rates = model_costs[model_name]
    return rates["input"] * input_tokens + rates["output"] * output_tokens
