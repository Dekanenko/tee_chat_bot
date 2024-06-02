# Chatbot for TeeCustomizer app

Chatbot assists users with selecting styles, colors, sizes, and printing options, answering FAQs, and log support requests.

## Run application

Before launching an application it is important to set the LANGCHAIN_API_KEY and credentials for Firebase service.

### Lang Smith
[Create a project](https://smith.langchain.com) and create an API key in settings

### Firebase
[Create a project](https://console.firebase.google.com)
In the project create a collection (no matter which one just create), otherwise you could get an error.
Project Settings -> Generate a new private key
Upload the key into credentials folder and rename it to serviceAccountKey.json

Enable Firestore API if you didn't.

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
- Chatbot uses RAG to provide more accurate information and answer questions
- Log support requests. Model recognizes the user's struggles and capture details like issue and order number, storing them in database
- Chainlit interface for interaction
- Logging the runs from LLM using LangSmith