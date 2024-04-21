from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain
from langchain.chat_models import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from pyprojroot import here
from dotenv import load_dotenv
import os
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import langchain
langchain.debug = True

print(load_dotenv())

db_path = str(here("data")) + "/sqldb.db"
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

model_name = os.getenv("gpt_deployment_name")
azure_openai_api_key = os.environ["OPENAI_API_KEY"]
azure_openai_endpoint = os.environ["OPENAI_API_BASE"]
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_deployment=model_name,
    model_name=model_name,
    temperature=0.0)

chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
print(response)
db.run(response)
chain.get_prompts()[0].pretty_print()

execute_query = QuerySQLDataBaseTool(db=db)
write_query = create_sql_query_chain(llm, db)
chain = write_query | execute_query
print(chain.invoke({"question": "How many employees are there"}))

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

answer = answer_prompt | llm | StrOutputParser()
chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer
)

print(chain.invoke({"question": "How many employees are there"}))
