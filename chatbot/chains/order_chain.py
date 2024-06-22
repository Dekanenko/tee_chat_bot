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
            stop=["user:"] 
        )

# create prompt
#     feature_name: [put one of the following: undefined, style, gender, color, size, print_option, preferences]
        template = """
Your task is to extract a provided feature from user's input. Extract information precisely, do not change that!
If the user's input does not match with the given feature (e.g., style - red), output "-".
Your responses should strictly follow the format below:
    feature_value: [put the corresponding value that would fit the feature; put "-" otherwise]

Examples:

Extract style.
User: red.
Assistant: {{
    "feature_value": "-"
}}

Extract size.
User: M.
Assistant: {{
    "feature_value": "M"
}}

Extract print_option.
User: Embroidery.
Assistant: {{
    "feature_value": "Embroidery"
}}

Extract style.
User: Tank Top.
Assistant: {{
    "feature_value": "Tank Top"
}}

Extract color.
User: blue.
Assistant: {{
    "feature_value": "blue"
}}

Extract style.
User: crew neck.
Assistant: {{
    "feature_value": "crew neck"
}}

Extract print_option.
User: screen printing.
Assistant: {{
    "feature_value": "screen printing"
}}

Extract color.
User: Not a color.
Assistant: {{
    "feature_value": "-"
}}

Extract size.
User: purple.
Assistant: {{
    "feature_value": "-"
}}

Extract print_option.
User: Heat Transfer.
Assistant: {{
    "feature_value": "Heat Transfer"
}}

User: {question}
Extract {order_feature} from user's input.
Assistant:

"""
        self.prompt = PromptTemplate(input_variables=["question", "order_feature"], template=template)

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
            return "-"
            
        return response_json['feature_value']

    def submit_order(self, db):
        db.collection('orders').add(self.order)

    def process_input(self, user_input, db):
        if self.order_counter == -1:
            self.order = {"style":"", "gender":"", "color":"", "size":"", "print_option":"", "preferences":"", "order_id":0}
            self.order_counter += 1
        else:
            result = self.chain({"question": user_input, "order_feature": self.order_keys[self.order_counter]})
            feature_value = self.response_parser(result)

            if "-" == feature_value:
                out = "Sorry, there might be a mistake😭. Could you please answer again to the following question:\n"+self.questions[self.order_counter]
                return out

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