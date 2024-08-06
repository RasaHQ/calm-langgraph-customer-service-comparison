# Customer Support Bot in Langgraph

The code for this bot is taken from 
[this official tutorial](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/) available on langgraph's documentation.
The objective here is to be able to evaluate the implementation on a few metrics - 

1. Cost
2. Latency per turn
3. Correctness

## Setup

Install the necessary packages with - 

```commandline
pip install requirements.txt
```

Setup the necessary env variables with API keys - 

1. `TAVILY_API_KEY` - Access key for Tavily, used for making search queries
2. `LANGCHAIN_API_KEY` - Langsmith access key, for monitoring and tracing LLM calls.
3. `OPENAI_API_KEY` - API key for OpenAI platform, for invoking the LLM.

Setup the database with - 
```commandline
cd ../
python scripts/create_db.py
```

## Evaluation

For the purpose of the evaluation, a bunch of test conversations have been written with varying complexity as defined [here](../README.md#quantitative-evaluation)

To run the tests, execute the command - 

```commandline
python run_eval.py
```

Once the script finishes you will see runtime stats on input and output tokens consumed and latency incurred. These stats are grouped by the folder which contained the tests - 

```commandline
Runtime stats for ../e2e_tests/context_switch
==============================
Input tokens stats:
Min: 720
Mean: 2490.625
Median: 2256.5
Max: 5735
------------------------------
Output tokens stats:
Min: 16
Mean: 75.875
Median: 57.0
Max: 174
------------------------------
Latency stats:
Min: 2.8925740718841553
Mean: 6.08220374584198
Median: 4.617524027824402
Max: 10.86777400970459
------------------------------
```

Also, the predicted conversations are output to the file `predictions.yaml` in the same E2E format that we use for writing input tests. Viewing this file would help you to see where the bot struggles to help the user.

### Customization

#### Changing the test set

You can change the set of test conversations that are run by excluding / including a folder containing E2E test conversations.
To do so, modify the following list in `run_eval.py`

```python

test_folders = [
            "../e2e_tests/happy_paths",
            "../e2e_tests/multi_skill",
            "../e2e_tests/corrections",
            "../e2e_tests/context_switch",
        ]
```

#### Changing the LLM being used

You can change the LLM used by changing line `49` of `app.py` - 

```python
model_name="gpt-4"
```

Set the model to [one of the model tags](https://platform.openai.com/docs/models) 
from OpenAI.


## TODOs

- [ ] Make output file configurable
- [ ] Make test folder customization easier
- [ ] Make changing the LLM easier
- [ ] Employ an automated way to evaluate the correctness of the bot, for e.g. via an LLM judge
