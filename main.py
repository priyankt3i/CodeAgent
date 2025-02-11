import subprocess
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler
import time

st.title("Code Analysis and Test Case Generator")

# User input for code and prompt
user_code = st.text_area("Enter your Python code", height=300)
user_prompt = st.text_area("Enter your prompt", height=150)

# Function to execute Python code safely
def execute_python_code(code):
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
def generate_test_cases():
    global user_code, user_prompt
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
    final_prompt = (
        "Generate multiple test cases for the following Python code "
        "and provide test data for all scenarios. Do not add any heading, comments, or explanations. Always respond in json and json only format with tags testName, testData, expectedResult:\n"
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
        st.write(f"Parsed test cases: {test_cases}")  # Debug print
    except json.JSONDecodeError as e:
        st.write(f"Invalid JSON: {e}")
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
                st.write(f"Skipping invalid test case: {case}")
    else:
        st.write("Error: test_cases is not a list!")
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

# Streamlit App
def run_streamlit_app():
    if st.button("Run"):
        if user_code and user_prompt:
            # Step 1: Analyze Code
            percentage_text = st.empty()  # This will hold the percentage label
            st.subheader("Step 1: Analyze Code")
            progress_bar = st.progress(10)
            percentage_text.write("10%")
            analysis = agent.run(f"Analyze this code: {user_code}")
            progress_bar.progress(33)
            percentage_text.write("33%")
            st.write(f"Analysis: {analysis}")

            # Show intermediary thoughts using StreamlitCallbackHandler
            st.write("Chain of thoughts: ")
            st_cb = StreamlitCallbackHandler(st.container())  # Capture output in the container
            try:
                result = agent.invoke({"input": f"Analyze this code: {user_code}"}, callbacks=[st_cb])
                st.write(f"Analysis outcome: {result['output']}")
            except Exception as ex:
                st.exception(ex)

            # Step 2: Generate Test Cases
            st.subheader("Step 2: Generate Test Cases")
            progress_bar.progress(66)
            percentage_text.write("66%")
            test_cases = generate_test_cases()  
            progress_bar.progress(100)
            percentage_text.write("100%")        
            # Step 3: Validate Code
            st.subheader("Step 3: Validate Code")
            try:
                validation_results = validate_code(user_code, test_cases)
                st.write(f"Validation Results: {validation_results}")
            except json.JSONDecodeError as e:
                st.error(f"Error decoding test cases: {e}")
        else:
            st.error("Please provide both Python code and a prompt!")

if __name__ == "__main__":
    run_streamlit_app()