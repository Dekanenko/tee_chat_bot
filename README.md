# Chatbot for TeeCustomizer platform

Chatbot assists users with selecting styles, colors, sizes, and printing options. It also answers FAQs and logs requests.

## Run application

Before launching an application it is important to set the LANGCHAIN_API_KEY and credentials for Firebase service.

### Lang Smith
[Create a project](https://smith.langchain.com) and create an API key in settings.

### Firebase
[Create a project](https://console.firebase.google.com)
- In the project create a collection (no matter which one just create), otherwise you could get an error
- Project Settings -> Generate a new private key
- Upload the key into credentials folder and rename it to serviceAccountKey.json
- Enable Firestore API if you didn't

Also you should switch to Firestore Native mode:
1. Go to the Firebase Console.
2. Select your project.
3. In the left-hand menu, select Firestore Database.
4. Check if the database is in Datastore Mode or Native Mode.

You are ready to go🥳

```
chainlit run app.py
```

## Features
- 3 chains were developed to make a Chatbot perform different tasks:
    1st Chain answers user's questions and logs support requests for them. It recognizes users' struggles and direct requests, creating a support_request document and storing it in DB
    2nd Chain gives the user a list of questions and parses its input, filling out the order form. Once the form is finished and the user's confirmation is obtained, the order is stored in DB
    3rd Chain orchestrates input data. Once a user's input is obtained, this model decides where it should go: to the 1st chain or the 2nd. 
- Chatbot uses RAG and memory to provide more accurate information and answer questions
- Chainlit interface for interaction
- Logging the runs from LLM using LangSmith