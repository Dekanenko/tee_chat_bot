import json
from huggingface_hub import hf_hub_download
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from chatbot.data_preparation import get_vecdb

class Chat_bot():

    def __init__(self):

# prepare the data for context
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vecdb = get_vecdb(embedding_function)

# create llm
        model_name = "bartowski/Meta-Llama-3-8B-Instruct-GGUF"
        model_file = "Meta-Llama-3-8B-Instruct-IQ2_M.gguf"
        model_path = hf_hub_download(model_name, filename=model_file)
        n_gpu_layers = -1 
        n_batch = 32 

        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=True,
            temperature=0.1,
            stop=["Q", "Question", "User:", "Answer", "User's question", "Helpful Answer", "Unhelpful Answer", "Final Answer:", "Chat History", "Use the following pieces of context", "Other Helpful", "```", " } }"]
        )

# create prompt
        template = """
You are a chatbot on a virtual platform selling T-Shirts called TeeCustomizer.
When asked a question, first search the knowledge base for the most relevant question-answer pair.
Please answer the user's question in json format and include a support_request attribute if the question is related to a support issue. Keep the answer as concise as possible.

Example:
User: I need help with my order. My order number is 12345 and it hasn't arrived yet.
Assistant: {{
    "response": "I'm sorry to hear that your order hasn't arrived yet. I'll log a support request for you.",
    "support_request": {{
        "issue": "Order not arrived",
        "order_number": "12345"
    }}
}}

Chat History:
{chat_history}
    
User:{question}
Assistant:
"""
        self.prompt = PromptTemplate(input_variables=["question", "chat_history"], template=template)
        
# create chain
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  
            output_key="answer",
            ai_prefix="Assistant",
            human_prefix="User",
            return_docs=False,
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm, 
            chain_type="stuff", 
            retriever=self.vecdb.as_retriever(search_type="similarity_score_threshold", 
                                              search_kwargs={"score_threshold": 0.05, "k": 4}), 
            condense_question_prompt=self.prompt,
            memory=self.memory,
            return_source_documents=True,
            return_generated_question=True,
            get_chat_history=lambda h : ""
        )

    def response_parser(self, response, db):
        if "response" in response['answer'] and "}" in response['answer']:
            try:
                response_json = json.loads(response['answer'])
                if "support_request" in response['answer']:
                    support_request = response_json.get("support_request")
                    db.collection('supportRequests').add(support_request)
                    
            except json.JSONDecodeError:
                return "There was an error parsing the response. Please try again."
            
            return response_json['response']
        else:
            return response['answer']

    def process_input(self, user_input, db):
        result = self.chain({"question": user_input})
        response = self.response_parser(result, db)
        return response

