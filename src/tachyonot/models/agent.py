import getpass
import os
from groq import Groq
from tachyonot.utils import config as tconfig
from tachyonot.utils.helpers import read_xlsx_file
from tachyonot.utils.templates import CSV_TEMPLATE, POST_CSV_TEMPLATE

class Agent:
    def __init__(self, df):
        # self._initialize_model()
        self._intialize_groq()
        self.df = df

    def _initialize_model(self):
        self.llm = Llama(
            model_path=str(tconfig.model_path),
            temperature=0.1,
            max_new_tokens = 2048,
            n_ctx = 2048,
            n_threads = 1,
            verbose = False,
        )

    def _intialize_groq(self):
        self.groq = Groq(api_key=os.environ["GROQ_API_KEY"])


    def _generate_response(self, prompt):
        response = self.llm(prompt, max_tokens=500, stop=["```", "\n\n"], echo=False)
        return response['choices'][0]['text'].strip()

    def _generate_groq_response(self, query):
        response = self.groq.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": CSV_TEMPLATE.format(df_details=self.df.head(), query=query)
                }
            ],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content

    def respond(self, query):
        df = self.df
        processed_value = eval(self._generate_groq_response(query))
        print(processed_value)
        response = self.groq.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": POST_CSV_TEMPLATE.format(df_response=processed_value, query=query)
                }
            ],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your GROQ API Key: ")
    df = read_xlsx_file("CSV_FILE_PATH")
    agent = Agent(df)
    query = """
    A furnace is being is ued to heat a specific set of gas. 
    Give the logs file which contains data of gas pressure and temperature, what can you comment about the statistics of the gas pressure such as it's details?"  
    """
    print(agent.respond(query))
