# Rasa reimplementation of LangGraph's customer service bot

This is a reimplementation of LangGraph's [customer support](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/) example using [Rasa Pro](https://rasa.com/docs/rasa/), with a quantitative comparison of the two approaches.

There's a [YouTube video](https://www.youtube.com/watch?v=b3XsvoFWp4c) that walks through LangGraph's implementation.

> **Repo rename:** This project is intended to live at `rasa-langgraph-customer-service-comparison` (formerly `calm-langgraph-customer-service-comparison`). Until the GitHub rename is applied, clone URLs may still use the old name; GitHub will redirect after rename.

## Skills

The bot has the following skills:

* Showing the user's booked flights
* Changing flight bookings
* Booking a rental car
* Booking a hotel
* Booking an excursion (e.g. museum visit) on the trip

## Requirements

* **Rasa Pro 3.17** (evaluated with **3.17.2**)
* A [Rasa Pro license](https://rasa.com/docs/rasa-pro/installation/python/licensing/)
* An [OpenAI API key](https://platform.openai.com/api-keys)
* Familiarity with Python and Bash

## Running in a GitHub Codespace

Follow these steps to set up and run the Rasa assistant in a GitHub Codespace.

### Steps to run the Rasa assistant

To run the Rasa assistant, you can watch the video below and/or follow the instructions:

https://github.com/user-attachments/assets/45a828fe-b638-4d5f-8d55-22a831d5f198

1. **Create a Codespace:**

   - Navigate to the repository on GitHub.
   - Click on the green "Code" button, then scroll down to "Codespaces".
   - Click on "Create codespace on main branch".
   - This should take under two minutes to load.

  ![Screenshot 2024-08-05 at 11 01 26 AM](https://github.com/user-attachments/assets/0795f628-afb5-4d26-980a-807a2805169b)

2. **Set Up Environment:**
   - Once the Codespace loads, it will look like VS Code but in your browser!
   - Open a terminal and run `source .venv/bin/activate` to activate the development environment.
   - Open `/calm_llm/.env` and add the required keys:
     ```
     export RASA_PRO_LICENSE='your_rasa_pro_license_key_here'
     export OPENAI_API_KEY='your_openai_api_key_here'
     ```
   - Load the environment variables:
     ```
     source calm_llm/.env
     ```

3. **Create the Database:**
   - Run: `python scripts/create_db.py`.

4. **Train the Model:**
   - Enter the project directory, `cd calm_llm`, then run:
     ```
     rasa train
     ```

5. **Launch the Rasa Inspector:**
   - Once the model is trained, run:
     ```
     rasa inspect --debug
     ```

6. **Access the Inspector:**
   - When prompted to open in browser, click the link.

7. Chat with your customer support assistant about flights, hotels, cars, and/or excursions!

### Notes

- Keyboard bindings may not map correctly in the Codespace, so you may not be able to copy and paste as you normally would!
- The database creation is done separately to manage memory usage.
- This repository targets **Rasa Pro 3.17** (Codespace / `devcontainer` installs `rasa-pro==3.17.2`).
- Subdirectories:
  - `calm_llm` ù primary Rasa Pro LLM assistant used for the comparison
  - `calm_nlu` ù Rasa with intent-based NLU
  - `langgraph_implementation` ù LangGraph-style implementation inspired by [LangGraph's tutorial](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/)
  - `calm_self_hosted` ù Rasa with a fine-tuned self-hosted model (e.g. Llama 3.1 8B) as the command generator
  - `calm_nlu_self_hosted` ù Rasa with intent-based NLU and a fine-tuned self-hosted command generator

## Quantitative Evaluation

We provide scripts to evaluate the assistant on three measures:

* number of tokens used per user turn (proxy for LLM cost per user turn)
* latency (time to get a response)
* accuracy

The shared e2e suite covers:

* Happy paths ù conversations with minimal complexity sticking to one skill
* Slot corrections ù the user changes their mind mid-flow
* Context switches ù switch from one skill to another
* Cancellations ù the user stops mid-skill
* Multi skill ù several skills one after another

On **Rasa Pro 3.17.2**, `calm_llm` uses `SearchReadyLLMCommandGenerator` (via a thin token-tracking wrapper). Flow collect/branching fixes in this update were validated at **14/14** with the previous single-step generator on **gpt-4o** and **gpt-5.2**; re-validate after the SearchReady migration before treating that score as current.

### Run end-to-end tests

Ensure you have set up the environment by following [Steps to run the Rasa assistant](#steps-to-run-the-rasa-assistant).

In the `calm_llm` directory:

```
MAX_NUMBER_OF_PREDICTIONS=50 python run_eval.py
```

This prints results to the terminal. You can also pipe to a file:

```
MAX_NUMBER_OF_PREDICTIONS=50 python run_eval.py > results.txt
```

Once the script finishes you will see runtime stats on input and output tokens and latency, grouped by the folder that contained the tests:

```commandline
Running tests from ./e2e_tests/happy_paths
=============================
COST PER USER MESSAGE (USD)
---------------------------------
Mean: 0.031122631578947374
Min: 0.026789999999999998
Max: 0.038040000000000004
Median: 0.03162
---------------------------------

COMPLETION TOKENS PER USER MESSAGE
---------------------------------
Mean: 10.368421052631579
Min: 6
Max: 26
Median: 9.0
---------------------------------

PROMPT TOKENS PER USER MESSAGE
---------------------------------
Mean: 1016.6842105263158
Min: 881
Max: 1248
Median: 1021.0
---------------------------------

LATENCY PER USER MESSAGE (sec)
---------------------------------
Mean: 2.567301022379022
Min: 1.5348889827728271
Max: 4.782747983932495
Median: 2.067293882369995
---------------------------------

============================================================== short test summary info ===============================================================
================================================================= 0 failed, 5 passed
==================================================================
```

### LangGraph assistant

Navigate to `langgraph_implementation` and set up the environment:

```
# Step 1: Create a new virtual environment
python -m venv new_env

# Step 2: Activate the virtual environment
source new_env/bin/activate

# Step 3: Install the packages from requirements.txt
pip install -r requirements.txt
```

Open the `.env` file in that folder and set:

```
TAVILY_API_KEY - Access key for Tavily, used for making search queries
LANGCHAIN_API_KEY - Langsmith access key, for monitoring and tracing LLM calls.
OPENAI_API_KEY - API key for OpenAI platform, for invoking the LLM.
```

Load the keys with `source .env`, then run:

```
python run_eval.py
```

You can also pipe results: `python run_eval.py > results.txt`.

### Run tests to create figures

To create the figures in our blog post:

1. Generate data for the Rasa assistant

- Follow steps 2ù5 from [Steps to run the Rasa assistant](#steps-to-run-the-rasa-assistant).
- On a separate terminal, navigate to `calm_llm`, run `python run_tests_for_plots.py`.
- Restructure the data for plotting with `cd results` then `python combine_data.py`.

2. Generate data for the Rasa + NLU assistant

- Same as above, but in steps 4 and 5 use `cd calm_nlu` instead of `cd calm_llm`.
- On a separate terminal, navigate to `calm_nlu`, run `python run_tests_for_plots.py`.
- Restructure with `cd results` then `python combine_data.py`.

3. Generate data for the LangGraph assistant

- Follow the [LangGraph assistant](#langgraph-assistant) setup above.
- In `langgraph_implementation`, run `python run_tests_for_plots.py`.
- Restructure with `cd results` then `python combine_data.py`.

### Load and visualize results in a Jupyter Notebook

- Open `metrics.ipynb` (in the root directory)
- In the top-right of your screen, you should see 'Select Kernel', click on it
![Screenshot 2024-08-07 at 11 50 29 AM](https://github.com/user-attachments/assets/1e05d418-81b0-424d-b9f8-ca62234499cd)
- Once prompted, install necessary extensions
![Screenshot 2024-08-07 at 11 43 56 AM](https://github.com/user-attachments/assets/6562a48c-4275-4faa-8be1-c091c35bb529)
- Once the extensions are installed, click "select Kernel" again and select 'Python Environments...'
![Screenshot 2024-08-07 at 11 45 09 AM](https://github.com/user-attachments/assets/0bd26925-3a8e-45e9-8b1d-bdf1b40bf9ca)
- select the `.venv` environment for running the kernel:
![Screenshot 2024-08-07 at 11 45 31 AM](https://github.com/user-attachments/assets/6ce5581d-c470-40ed-9a1d-2a0dec4c4b06)
- execute all cells!
