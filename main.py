from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import PythonREPLTool, Tool

load_dotenv()

if __name__ == "__main__":
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    python_agent_executor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    # python_agent_executor.run(
    #     """generate and save in current working directory 15 QRcodes
    #                             that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    # )

    csv_agent_executor = create_csv_agent(
        llm=llm,
        path="./episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""useful when you need to transform natural language and write from it python and execute the python code,
                              returning the results of the code execution,
                            DO NOT SEND PYTHON CODE TO THIS TOOL""",
            ),
            Tool(
                name="CSVAgent",
                func=csv_agent_executor.raise_callback_manager_deprecation,
                description="""useful when you need to answer question over episode_info.csv file,
                             takes an input the entire question and returns the answer after running pandas calculations""",
            ),
        ],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )
    grand_agent.run(
        """generate and save in current working directory 15 QRcodes
                                that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    )

    grand_agent.run("print seasons ascending order of the number of episodes they have")
