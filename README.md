# CALM Reimplementation of LangGraph's customer service bot without OpenAI models

This branch includes instructions for running [CALM Reimplementation of LangGraph's customer service bot](https://github.com/RasaHQ/calm-langgraph-customer-service-comparison) without utilizing OpenAI models.

*Disclaimer: this implementation uses smaller models that have restrictions on the input token size. You might experience different behaviour from your assistant compared to bigger, e.g. OpenAI, models. If you'd like to run this setup in production, it is recommended to fine-tune and deploy the model in your own dedicated instance.*

## Why consider using custom LLM models?

- **Cost-effectiveness**
By default, CALM uses OpenAI models for powering LLM-based components like the Command Generator and Contextual Response Rephraser. While powerful OpenAI models provide great performance results, they can be quite costly when running assistants in production. The modular nature of CALM enables developers to use smaller, more cost-effective models that can be fine-tuned for specific tasks. Often, these models provide great performance results while keeping the cost low.
  
- **Customization and availability**
By using custom, open-source LLMs, developers can avoid vendor lock-in and rate limits imposed by third-party LLM providers.


## Skills

The bot has the following skills:
* Showing the user's booked flights
* Changing flight bookings
* Booking a rental car
* Booking a hotel
* Booking an excursion (e.g. museum visit) on the trip

## Running in a Github Codespace

Follow these steps to set up and run the Rasa assistant in a GitHub Codespace.

### Prerequisites

- You'll need a [Rasa Pro license](https://rasa.com/docs/rasa-pro/installation/python/licensing/).
- You should be familiar with Python and Bash.

### Steps to run the CALM assistant

This guide will show how you can run the CALM assistant that is configured to use Rasa's fine-tuned [CodeLlama 13B](https://huggingface.co/rasa/cmd_gen_codellama_13b_calm_demo) model instead of the OpenAI models. If you'd like to see an example of the same implementation using OpenAI models, switch to the `main` branch.

*Disclaimer: Since CodeLlama 13B is a much smaller than OpenAI models, you might noticed some performance differences. However, the performance of the models can be improved through fine-tuning.*

1. **Create a Codespace:**

   - Navigate to the repository on GitHub.
   - Click on the green "Code" button, then scroll down to "Codespaces".
   - Click on "Create codespace on no-openai-config branch".
   - This should take under two minutes to load.

  


2. **Set Up Environment:**
   - Once the Codespace loads, it will look like VSCode but in your browser!
   - Open a terminal and run `source .venv/bin/activate`
   - Open `/calm_llm/.env` file and add the required keys to that file.
     ```
     export RASA_PRO_LICENSE='your_rasa_pro_license_key_here'
     ```
   - Set your environment variables by running:
     ```
     source calm_llm/.env
     ```

3. **Create the Database:**
   - Run the command to create the database: `python scripts/create_db.py`.


4. **Train the Model:**
   - Enter the project directory, `cd calm_llm`, and then run:
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
- The repository is compatible with Rasa Pro versions `>=3.11.0rc1`.
- You'll also notice that there are several subdirectories: `calm_llm` is the CALM implementation with fine-tuned model, `calm_nlu` combines CALM with intent based NLU, `langgraph_implementation` is the implementation inspired from [langgraph's tutorial](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/), `calm_self_hosted` is the CALM implementation but a fine-tuned model such as Llama 3.1 8B working as the command generator, and `calm_nlu_self_hosted` is CALM working with intent based NLU and a fine-tuned model as the command generator.

## Quantitative Evaluation

We provide scripts to evaluate the assistant on 3 measures:
* number of tokens used per user turn (proxy for measuring LLM cost per user turn)
* latency (time to get a response)
* accuracy

To do so, we construct a test set to evaluate the following capabilities:
* Happy paths - Conversations with minimal complexity sticking to one skill.
* Slot corrections - Conversations where a user changes their mind in between and corrects themselves.
* Context switches - Conversations with a switch from one skill to another and coming back to the former skill.
* Cancellations - Conversations where the user decides to not proceed with the skill and stops midway.
* Multi Skill - Conversations where the user tries to accomplish multiple skills one after the other.

### Run end-to-end tests

Ensure you have set up the environment in two active terminals by following the instructions in [this section](#steps-to-run-the-calm-assistant)

Execute the following in the `calm_llm` directory:

```
MAX_NUMBER_OF_PREDICTIONS=50 python run_eval.py
```

This will print the results to your terminal. You can also pipe the results to a text file `MAX_NUMBER_OF_PREDICTIONS=50 python run_eval.py > results.txt`.

Once the script finishes you will see runtime stats on input and output tokens consumed and latency incurred. These stats are grouped by the folder which contained the tests - 

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

Navigate to `langgraph_implementation` folder and then set up the environment with - 

```
# Step 1: Create a new virtual environment
python -m venv new_env

# Step 2: Activate the virtual environment
source new_env/bin/activate

# Step 3: Install the packages from requirements.txt
pip install -r requirements.txt
```

Next, set up the necessary keys by opening the `.env` file in that folder and filling the values for requested variables 

```
TAVILY_API_KEY - Access key for Tavily, used for making search queries
LANGCHAIN_API_KEY - Langsmith access key, for monitoring and tracing LLM calls.
OPENAI_API_KEY - API key for OpenAI platform, for invoking the LLM.
```

Load the keys by running `source .env` in the terminal window.


Then execute -
```
python run_eval.py
```

This will print the results to your terminal. You can also pipe the results to a text file `python run_eval.py > results.txt`.

### Run tests to create figures

To create the figures in our blog post

1. Generate data for CALM assistant

- Follow steps 2-5 from [Steps to run CALM assistant](#steps-to-run-the-calm-assistant) section.
- On a separate terminal, navigate to `calm_llm` directory, run `python run_tests_for_plots.py` to generate data for figures.
- Restructure the data for plotting with `cd results` and then `python combine_data.py`


2. Generate data for CALM + NLU assistant

- Follow steps 2-5 from [Steps to run CALM assistant](#steps-to-run-the-calm-assistant) section but in Steps 4 and 5, `cd calm_nlu` instead of `cd calm_llm`.
- On a separate terminal, navigate to `calm_nlu` directory, run `python run_tests_for_plots.py` to generate data for figures
- Restructure the data for plotting with `cd results` and then `python combine_data.py`

3. Generate data for LangGraph assistant

- Run steps 1-5 from [LangGraph assistant](#langGraph-assistant) above
- In `langgraph_implementation` folder, run `python run_tests_for_plots.py` to generate data for figures
- Restructure the data for plotting with `cd results` and then `python combine_data.py`


### Load and visualize results in a Jupyter Notebook

- Open `metrics.ipynb` (in root directory)
- In the top-right of your screen, you should see 'Select Kernel', click on it
![Screenshot 2024-08-07 at 11 50 29 AM](https://github.com/user-attachments/assets/1e05d418-81b0-424d-b9f8-ca62234499cd)
- Once prompted, install necessary extensions
![Screenshot 2024-08-07 at 11 43 56 AM](https://github.com/user-attachments/assets/6562a48c-4275-4faa-8be1-c091c35bb529)
- Once the extensions are installed, click "select Kernel' again and select 'Python Environments...'
![Screenshot 2024-08-07 at 11 45 09 AM](https://github.com/user-attachments/assets/0bd26925-3a8e-45e9-8b1d-bdf1b40bf9ca)
- select the `.venv` environment for running the kernel:
![Screenshot 2024-08-07 at 11 45 31 AM](https://github.com/user-attachments/assets/6ce5581d-c470-40ed-9a1d-2a0dec4c4b06)
- execute all cells!





