import json
from chatbot.chains.orchestrator_chain import Orchestrator
from chatbot.chains.question_logs_chain import QuestionLogs
from chatbot.chains.order_chain import OrderChain

class Chat_bot():

    def __init__(self):
        self.orchestrator = Orchestrator()
        self.question_logs_chain = QuestionLogs()
        self.order_chain = OrderChain()
        
    def process_input(self, user_input, db):
        chain = self.orchestrator.process_input(user_input)
        if chain == 1:
            result = self.question_logs_chain.process_input(user_input, db)
        if chain == 2:
            result = self.order_chain.process_input(user_input, db)
        
        return result