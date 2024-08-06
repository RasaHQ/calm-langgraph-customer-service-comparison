# CALM Reimplementation of LangGraph's customer service bot

This is a reimplementation in CALM of langgraph's [customer support](https://langchain-ai.github.io/langgraph/tutorials/chatbots/customer-support/) example. 
There's a [YouTube video](https://www.youtube.com/watch?v=b3XsvoFWp4c) that provides a walkthrough. 


## Skills

The bot has the following skills:
* Showing the user's booked flights
* Changing flight bookings
* Booking a rental car
* Booking a hotel
* Booking an excursion (e.g. museum visit) on the trip
* Answering questions about SWISS AIR's ticket policies (TODO: Not implemented yet)

## Running in a Github Codespace

Follow these steps to set up and run the Rasa assistant in a GitHub Codespace.

### Prerequisites

- You must have access to the private repository.
- You'll need a [Rasa Pro license](https://rasa.com/docs/rasa-pro/installation/python/licensing/) and an [OpenAI API key](https://platform.openai.com/playground/chat).
- You should be familiar with Python and Bash.

### Steps to run CALM assistant

1. **Create a Codespace:**

   - Navigate to the repository on GitHub.
   - Click on the green "Code" button, then scroll down to "Codespaces".
   - Click on "Create codespace on main branch".
   - This should take under two minutes to load.

  ![Screenshot 2024-08-05 at 11 01 26 AM](https://github.com/user-attachments/assets/0795f628-afb5-4d26-980a-807a2805169b)


2. **Set Up Environment:**
   - Once the Codespace loads, it will look like VSCode but in your browser!
   - First, open two terminal windows.
   - In both terminals, run `source .venv/bin/activate`
   - Open `/calm_llm/.env` file and add the required keys to that file.
     ```
     export RASA_PRO_LICENSE='your_rasa_pro_license_key_here'
     export OPENAI_API_KEY='your_openai_api_key_here'
     ```
   - In both terminals, set your environment variables by running:
     ```
     source calm_llm/.env
     ```

3. **Create the Database:**
   - In one of the terminals, run the command to create the database: `python scripts/create.db`.


4. **Train the Model:**
   - In the first terminal, `cd calm_llm` and then run:
     ```
     rasa train
     ```


5. **Start the Action Server:**
   - In the second terminal, `cd calm_llm` and the run:
     ```
     rasa run actions
     ```

6. **Launch the Rasa Inspector:**
   - In the first terminal, run:
     ```
     rasa inspect --debug
     ```

7. **Access the Inspector:**
   - When prompted to open in browser, click the link.


8. Chat with your customer support assistant about flights, hotels, cars, and/or excursions!


### Notes

- Keyboard bindings may not map correctly in the Codespace so you may not be able to copy and paste as you normally would!
- The database creation is done separately to manage memory usage.

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

Execute the following in the `calm_llm` while the action server is running:

```
python run_eval.py
```

This will print the results to your terminal. You can also pipe the results to a text file `python run_eval.py > results.txt`.

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

Navigate to `langgraph_implementation` and then

```
# Step 1: Create a new virtual environment
python -m venv new_env

# Step 2: Activate the virtual environment
source new_env/bin/activate

# Step 3: Install the packages from requirements.txt
pip install -r requirements.txt
```

Step 4: Then set up the necessary keys by opening the `.env` file 

```
TAVILY_API_KEY - Access key for Tavily, used for making search queries
LANGCHAIN_API_KEY - Langsmith access key, for monitoring and tracing LLM calls.
OPENAI_API_KEY - API key for OpenAI platform, for invoking the LLM.
```

Step 5: Load the keys by running `source .env` in the terminal window.


Step 6: Then execute
```
python run_eval.py
```

This will print the results to your terminal. You can also pipe the results to a text file `python run_eval.py > results.txt`.

### Run tests to create figures

To create the figures in our blog post

1. Generate data for CALM assistant

- Follow steps 2-6 above in "Steps to run CALM assistant"
- In `calm_llm`, run `python run_tests_for_plots.py` to generate data for figures
- `cd results` and then `python combine_data.py` to restructure data for plotting

2. Generate data for CALM + NLU assistant

- Follow steps 2-6 above in "Steps to run CALM assistant"
- In `calm_nlu`, run `python run_tests_for_plots.py` to generate data for figures
- `cd results` and then `python combine_data.py` to restructure data for plotting

3. Generate data for LangGraph assistant

- Run steps 1-5 from [LangGraph assistant](#LangGraph-assistant) above
- In `langgraph_implementation` folder, run `python run_tests_for_plots.py` to generate data for figures
- `cd results` and then `python combine_data.py` to restructure data for plotting

### Load and visualize results in a Jupyter Notebook

- open `metrics.ipynb` (in root directory)
- In the top-right of your screen, you should see 'Select Kernel', click on it
- Once prompted, install necessary extensions
- Once the extensions are installed, select the `.venv` environment for running the kernel.
- execute all cells!





