import sqlite3
import streamlit as st
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.utilities.sql_database import SQLDatabase
import os
from langchain.agents import AgentType
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
import pandas as pd
import subprocess
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, AgentExecutor
from langchain.tools import Tool
from langchain.callbacks import StreamlitCallbackHandler
import time

from app import analyze_code, execute_python_code

st.set_page_config(page_title="Analyze Code", page_icon="🛢")
st.header('Analyze Code')
st.write('Code Analysis and Test Case Generator')
# User input for code and prompt
user_code = st.text_area("Enter your Python code", height=300)
user_prompt = st.text_area("Enter your prompt", height=150)

class run_streamlit_app:

    def __init__(self):
       if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

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

    def setup_agent(_self):
        # Initialize the agent
            # Define tools for the agent
        tools = [
            Tool(name="PythonExecutor", func=execute_python_code, description="Executes Python code and returns output."),
            Tool(name="CodeAnalyzer", func=analyze_code, description="Analyzes Python code for issues.")
        ]
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent    
    
    def main(self):

        agent = self.setup_agent()

        if st.button("Run"):
            if user_code and user_prompt:
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                st.chat_message("user").write(user_prompt)
                with st.chat_message("assistant"):
                    st_cb = StreamlitCallbackHandler(st.container())
                    try:
                        result = agent.invoke(
                            {"input": user_prompt},
                            {"callbacks": [st_cb]}
                        )
                        response = result["output"]
                        st.session_state.messages.append({"role": "assistant", "content": response})

                        if isinstance(response, pd.DataFrame):
                            st.info('Response is a Table')
                            df = pd.DataFrame(response)
                            st.dataframe(df.style.highlight_max(axis=0))
                        else:
                            #st.info('Response is not a Table')
                            st.success(response)

                    except Exception as ex:
                        st.exception(ex)


if __name__ == "__main__":
    obj =  run_streamlit_app()
    obj.main()