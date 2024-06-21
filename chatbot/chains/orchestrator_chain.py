import json
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from llama_cpp.llama import LlamaGrammar

from chatbot.data_preparation import get_vecdb

class Orchestrator():

    def __init__(self):
# create llm
        model_name = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
        model_file = "Meta-Llama-3-8B-Instruct-IQ2_M.gguf"
        model_path = hf_hub_download(model_name, filename=model_file)
        n_gpu_layers = -1 
        n_batch = 32 

        schema = r'''
        root ::= (
        "{" newline
            doublespace "\"chain\":" space number newline
        "}"
        )

        newline ::= "\n"
        space ::= " "
        doublespace ::= "  "
        number ::= [0-9]+
        '''

        grammar = LlamaGrammar.from_string(schema)

        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=True,
            temperature=0.1,
            grammar=grammar
        )

# create prompt
        template="""
Analyze the following input and output only one number: 1 or 2.
    If the user's input is a question (e.g., "What", "Which", "How", "Do you", "Can I") or an issue/request (e.g., "problem", "issue", "missing", "can't", "not working", "broken", "help", "order number"), output the number 1.
    If the user's input provides details for a T-shirt customization (e.g., color, size, design, etc.) or user says directly that they want to make order, output the number 2.
    If the user provides preferences for the order, output the number 2.

Use the following examples for better output:

User: What printing options do you offer?
Assistant: {{
    "chain": 1
}}

User: The t-shirt I ordered is missing from my delivery. My order number is 99876.
Assistant: {{
    "chain": 1
}}

User: Crew Neck, V-Neck, Long Sleeve, Tank Top; red, purple; male, unisex; M, XL; Screen Printing, Embroidery, Heat Transfer, Direct-to-Garment
Assistant: {{
    "chain": 2
}}

User: I want to customize a T-shirt
Assistant: {{
    "chain": 2
}}

User: I want the materials to be eco-friendly
Assistant: {{
    "chain": 2
}}
    
User:{question}
Assistant:
"""
        self.prompt = PromptTemplate(input_variables=["question"], template=template)

# create chain

        self.chain = LLMChain(
            llm = self.llm, 
            prompt = self.prompt
        )

    def response_parser(self, response):
        try:
            response_json = json.loads(response['text'])
            return True, response_json['chain']
        
        except json.JSONDecodeError:
            return False, "There was an error parsing the response. Please try again."
            
    def process_input(self, user_input):
        tries = 0
        correct = False
        while tries < 3 and not correct:
            result = self.chain({"question": user_input})
            correct, response = self.response_parser(result)
            tries += 1
        
        return response