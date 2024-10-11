TEMPLATE = """You are a helpful assistant and you have to answer questions according to the context given. 
Only reply with response. If you are unsure repond with 'I am not sure'.

{context}

Question: {question}"""

CSV_TEMPLATE = """
You are working with a pandas dataframe in Python. The name of the dataframe is df. You need to respond back with Python code that would return what the user wants. Your response should be executable Python code only, without any additional explanation or text.

Here are some examples:

User Query: Return the mean of the column 'gas_pressure' in the dataframe.
df['gas_pressure'].mean()

User Query: Return the first 5 rows of the dataframe.
df.head()

User Query: Return values of the column 'gas_pressure' which has its equivalent 'timestamp' column value greater than '2022-01-01 00:00:00'.
df[df['timestamp'] > '2022-01-01 00:00:00']['gas_pressure']

User Query: Return the unique values of the column 'gas_pressure'.
df['gas_pressure'].unique()

User Query: Return the rows where values of 'gas_pressure' column are greater than 100
df[df['gas_pressure'] > 100]

Here are the details of the dataframe which is obtained by using df.head():
{df_details}

The user query for the given dataframe is:
User Query: {query}
"""

POST_CSV_TEMPLATE = """
Give to you the user query and dataframe details (which is obtained from another robot) or a value describing what the user wants, 
explain in a professional conversational way to the user such that the answer corresponds to what the user wants and refers to the dataframe details.

Example:
User Query: Return the mean of the column 'gas_pressure' in the dataframe.
Dataframe Details or Value: 100.0
Answer: For the give logs data the mean value of the gas pressure is 100.0.

User Query: {query}
Dataframe Details or Value: {df_response}

Answer: 
"""

