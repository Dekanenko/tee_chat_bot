import json
from chatbot.chains.orchestrator_chain import Orchestrator

from chatbot.data_preparation import get_vecdb

class Chat_bot():

    def __init__(self):
        self.orchestrator = Orchestrator()

    # def response_parser(self, response, db):
    #     try:
    #         response_json = json.loads(response['answer'])
    #         if response_json["support_request"]["order_id"] != 0:
    #             db.collection('support_requests').add(response_json["support_request"])
                    
    #     except json.JSONDecodeError:
    #         return False, "There was an error parsing the response. Please try again."
            
    #     return True, response_json['response']

    def process_input(self, user_input, db):
        result = self.orchestrator.process_input(user_input)
        print(f"\n\n{result}\n\n")

        
        return result