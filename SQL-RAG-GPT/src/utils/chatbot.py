import os
from typing import List, Tuple
from utils.load_config import LoadConfig
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent

import langchain
langchain.debug = True

APPCFG = LoadConfig()
URL = "https://github.com/Farzad-R/LLM-Zero-to-Hundred/tree/master/sql-csv-RAG-GPT"
hyperlink = f"[SQL-CSV-RAG-GPT user guideline]({URL})"


class ChatBot:
    """
    Class representing a chatbot with document retrieval and response generation capabilities.

    This class provides static methods for responding to user queries, handling feedback, and
    cleaning references from retrieved documents.
    """
    @staticmethod
    def respond(chatbot: List, message: str, data_type: str, chatbot_functionality: str) -> Tuple:
        if chatbot_functionality == "Chat":
            if data_type == "Preprocessed SQL-DB":
                # directories
                if os.path.exists(APPCFG.sqldb_directory):
                    db = SQLDatabase.from_uri(
                        f"sqlite:///{APPCFG.sqldb_directory}")
                    execute_query = QuerySQLDataBaseTool(db=db)
                    write_query = create_sql_query_chain(APPCFG.llm, db)
                    answer_prompt = PromptTemplate.from_template(
                        APPCFG.llm_system_role)
                    answer = answer_prompt | APPCFG.llm | StrOutputParser()
                    chain = (
                        RunnablePassthrough.assign(query=write_query).assign(
                            result=itemgetter("query") | execute_query
                        )
                        | answer
                    )
                    response = chain.invoke({"question": message})

                else:
                    chatbot.append(
                        (message, f"SQL DB does not exist. Please first create the 'sqldb.db'. For further information please visit {hyperlink}."))
                    return "", chatbot, None
            elif data_type == "Uploaded CSV/XLSX SQL-DB" or data_type == "Stored CSV/XLSX SQL-DB":
                if data_type == "Uploaded CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.uploaded_files_sqldb_directory):
                        engine = create_engine(
                            f"sqlite:///{APPCFG.uploaded_files_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                        print(db.dialect)
                    else:
                        chatbot.append(
                            (message, f"SQL DB from the uploaded csv/xlsx files does not exist. Please first upload the csv files from the chatbot. For further information please visit {hyperlink}."))
                        return "", chatbot, None
                elif data_type == "Stored CSV/XLSX SQL-DB":
                    if os.path.exists(APPCFG.stored_csv_xlsx_sqldb_directory):
                        engine = create_engine(
                            f"sqlite:///{APPCFG.stored_csv_xlsx_sqldb_directory}")
                        db = SQLDatabase(engine=engine)
                    else:
                        chatbot.append(
                            (message, f"SQL DB from the stored csv/xlsx files does not exist. Please first execute `src/prepare_csv_xlsx_db` module. For further information please visit {hyperlink}."))
                        return "", chatbot, None
                print(db.dialect)
                print(db.get_usable_table_names())
                agent_executor = create_sql_agent(
                    APPCFG.llm, db=db, agent_type="openai-tools", verbose=True)
                response = agent_executor.invoke({"input": message})
                response = response["output"]

            chatbot.append(
                (message, response))
            return "", chatbot
        else:
            pass
