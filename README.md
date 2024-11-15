# adaptive-pipeline-model-training-batch

## Running and Testing the Code Locally

To run and test the code locally, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/odegay/adaptive-pipeline-model-training-batch.git
   cd adaptive-pipeline-model-training-batch
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```

4. Set the environment mode to `TEST` in the `src/main.py` and `src/load_features.py` files:
   ```python
   ENV_MODE = "TEST"  # Change to "PROD" for production execution
   ```

5. Create the `TESTS/data` directory to store JSON files for database call emulation:
   ```bash
   mkdir -p TESTS/data
   ```

6. Add your local data JSON file to the `TESTS/data` directory. For example, create a file named `local_data.json` and add your data in JSON format.

7. Run the main script:
   ```bash
   python src/main.py
   ```

8. Check the logs to verify that the code is running and testing locally. Messages should be saved to the log instead of using Pub/Sub, and data should be loaded from and saved to the JSON file in the `TESTS/data` directory.
