import subprocess
import json
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool


# Example: User request
user_code = """def is_prime(n): 
    if n <= 1: 
        return False 
    for i in range(2, n): 
        if n % i == 0: 
            return False 
    return True"""

user_prompt = "Check if a number is prime and generate multiple test cases for all scenarios along with test data."

# Function to execute Python code safely
def execute_python_code(code):
    st.write(code)
    try:
        result = subprocess.run(["python", "-c", code], capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr
    except Exception as e:
        return str(e)


# Function to analyze code for issues
def analyze_code(code):
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    prompt = f"Analyze the following Python code for syntax, logic errors, and improvements:\n```python\n{code}\n```"
    return llm.predict(prompt)


# Function to generate test cases
def generate_test_cases():  # input_str is not used
    global user_code, user_prompt
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    final_prompt = (
        "Generate multiple test cases for the following Python code "
        "and provide test data for all scenarioo. Do not add any heading, comments, or explanations. Always respond in json and json only format with tags testName, testData, expectedResult:\n"
        f"User intent: {user_prompt}\n"
        "```python\n"
        f"{user_code}\n"
        "```"
    )
    response = llm.predict(final_prompt)
    try:
        # Strip unnecessary parts and parse the JSON
        json_data = response.strip('```').strip().strip('json\n')
        test_cases = json.loads(json_data)
        print(f"Parsed test cases: {test_cases}")  # Debug print
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return []  # Return an empty list if there's an error
    
    return test_cases

# Function to validate code with test cases
def validate_code(code, test_cases):
    results = {}
    if isinstance(test_cases, list):  # Check if test_cases is a list
        for case in test_cases:
            if isinstance(case, dict):  # Ensure each case is a dictionary
                inputs = case["testData"]
                expected_output = case["expectedResult"]
                test_code = f"{code}\nprint({inputs})"
                output = execute_python_code(test_code).strip()
                results[inputs] = {"expected": expected_output, "actual": output, "passed": output == expected_output}
            else:
                print(f"Skipping invalid test case: {case}")
    else:
        print("Error: test_cases is not a list!")
    return results


# Define tools for the agent
tools = [
    Tool(name="PythonExecutor", func=execute_python_code, description="Executes Python code and returns output."),
    Tool(name="CodeAnalyzer", func=analyze_code, description="Analyzes Python code for issues.")
]

# Initialize the agent
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Step 1: Analyze Code
print("Step 1: Analyze Code")
analysis = agent.run(f"Analyze this code: {user_code}")
print("Analysis:", analysis)

# Step 2: Generate Test Cases
print("# Step 2: Generate Test Cases")
test_cases = generate_test_cases()  # Pass a dummy input
print("Test Cases:", test_cases)

# Step 3: Validate Code
print("Step 3: Validate Code")
try:
    # Ensure that test_cases is valid JSON
    parsed_test_cases = test_cases
    validation_results = validate_code(user_code, parsed_test_cases)
    print("Validation Results:", validation_results)
except json.JSONDecodeError as e:
    print(f"Error decoding test cases: {e}")
