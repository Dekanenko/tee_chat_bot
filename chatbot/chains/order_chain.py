import json
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from llama_cpp.llama import LlamaGrammar
from langchain.chains import LLMChain
import random

from chatbot.data_preparation import get_vecdb

class OrderChain():

    def __init__(self):
# create llm
        model_name = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
        model_file = "Meta-Llama-3-8B-Instruct-IQ2_M.gguf"
        model_path = hf_hub_download(model_name, filename=model_file)
        n_gpu_layers = -1 
        n_batch = 32 

        schema = r'''
        root ::= (
        "{"newline
            doublespace "\"feature_value\":" space string newline
        "}"
        )
        newline ::= "\n"
        doublespace ::= "  "
        space ::= " "
        string ::= "\""   ([^"]*)   "\""
        '''

        grammar = LlamaGrammar.from_string(schema)

        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=True,
            temperature=0.1,
            grammar=grammar,
        )

# create prompt
#     feature_name: [put one of the following: undefined, style, gender, color, size, print_option, preferences]
        template = """
Your task is to extract a provided feature from user's input. Extract information precisely, do not change that!
Your responses should strictly follow the format below:
    feature_value: [put the corresponing value that would fit a feature_name; put - if feature_name is undefined]

User:{question}

Extract {order_question} from user's input
Assistant:
"""
        self.prompt = PromptTemplate(input_variables=["question", "order_question"], template=template)

# create chain
        self.chain = LLMChain(
            llm=self.llm, 
            prompt=self.prompt,
        )

        self.order_counter = -1
        self.questions = [
            """What style of T-shirt would you like? Please choose one of the following options:
- Crew Neck
- V-Neck
- Long Sleeve
- Tank Top
""",
            """Who will be wearing this T-shirt? Please select one of the following:
- Male
- Female
- Unisex""",
            """What color would you like your T-shirt to be? Please choose one of the following options:
- White
- Black
- Blue
- Red
- Green
- Custom Color (If custom, please specify the color)""",
            """What size do you need? Available sizes are:
- XS
- S
- M
- L
- XL
- XXL""",
            """How would you like the design to be printed on the T-shirt? Please select one of the following:
- Screen Printing
- Embroidery
- Heat Transfer
- Direct-to-Garment""",
            """Any other preferences?""",
        ]
        
        self.order_keys = ["style", "gender", "color", "size", "print_option", "preferences", "confirm/deny"]

    def response_parser(self, response):
        try:
            response_json = json.loads(response['text'])                    
        except json.JSONDecodeError:
            return False, "There was an error parsing the response. Please try again."
            
        return True, response_json['feature_value']

    def submit_order(self, db):
        db.collection('orders').add(self.order)

    def process_input(self, user_input, db):
        if self.order_counter == -1:
            self.order = {"style":"", "gender":"", "color":"", "size":"", "print_option":"", "preferences":"", "order_id":0}
            self.order_counter += 1
        else:
            result = self.chain({"question": user_input, "order_question": self.order_keys[self.order_counter]})

            _, feature_value = self.response_parser(result)

            if self.order_counter == len(self.order_keys)-1:
                self.order_counter = -1
                if "confirm" in feature_value or "yes" in feature_value:
                    order_id = random.randint(1, 99999)
                    self.order["order_id"] = order_id
                    self.submit_order(db)
                    return "Hurray🥳, you have successfully ordered a T-shirt\nYour order id: "+str(order_id)
                else:
                    return "Sory if I missed some information😖. Let's try again"

            self.order[self.order_keys[self.order_counter]] = feature_value
            
            self.order_counter += 1
            if self.order_counter == len(self.order_keys)-1:
                out = "Please, verify the order:\n"
                for key in self.order_keys[:-1]:
                    out += "- " + key + " : " + self.order[key] + "\n"
                out += "\n\nEnter confirm, if the order is correct, enter deny otherwise"
                return out
        
        return self.questions[self.order_counter]