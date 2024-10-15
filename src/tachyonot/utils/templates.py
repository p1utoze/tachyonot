TEMPLATE = """You are a helpful assistant and you have to answer questions according to the context given. 
Only reply with response. If you are unsure repond with 'I am not sure'.

{context}

Question: {question}"""

CSV_TEMPLATE = """
Your are a data scientist helping out user analyse a CSV or Excel file or a database. You are working with a pandas dataframe in Python. 

The name of the dataframe is df. You need to respond back with Python code that would return what the user wants. Your response should be executable Python code only, without any additional explanation or text.


Here are the details of the dataframe which is obtained by using df.head():
{df_details}

Do not provide the answer with natural words. Only panadas dataframe syantax such as df.describe(). If you are unable to provide answer to the user's query respond with df.describe().

For the given below user query what is your answer?
{query}
"""

POST_CSV_TEMPLATE = """
Give to you the user query and numerical value which is a single word reponse to user quetion, explain in a professional conversational way to the user such that the answer corresponds to what the user wants and refers to the numerical value.

For the given user query and the numerical value as a reponse to it, respond in a natual or conversational way.
The use query is "{query}" and the value is {df_response}
"""

