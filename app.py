import os
import pickle
import json
import openai
import psutil

from flask import Flask, render_template, request, jsonify
from openai.error import RateLimitError
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain import OpenAI
from llama_index import GPTVectorStoreIndex, download_loader, StorageContext, load_index_from_storage, LLMPredictor, ServiceContext
#from dotenv import load_dotenv
#load_dotenv()
#os.getenv(OPENAI_API_KEY)
OPENAI_API_KEY = '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'
app = Flask(__name__)
process = psutil.Process()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/gpt3', methods=['GET', 'POST'])
def gpt4():
    user_input = request.args.get('user_input') if request.method == 'GET' else request.form['user_input']
    messages = [{"role": "user", "content": user_input}]

    try:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        # load index
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query("Select up to 3 quizzes from the document that you think I would like to take based on this input:" + user_input)
        content = str(response)
        print(response.get_formatted_sources())
        print(process.memory_info().rss)
        if "documents provided" in content:
            content = "I could not find a relevant Expert Interview."
    except RateLimitError:
        content = "The server is experiencing a high volume of requests. Please try again later."

    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True)