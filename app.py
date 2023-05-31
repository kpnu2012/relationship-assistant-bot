import psutil
from flask import Flask, render_template, request, jsonify
from openai.error import RateLimitError
from llama_index import GPTVectorStoreIndex, download_loader, StorageContext, load_index_from_storage, LLMPredictor, ServiceContext

from llama_index import (
    GPTKeywordTableIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext
)
from langchain import OpenAI


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
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=1, model_name="text-davinci-002"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
        # load index
        index = load_index_from_storage(storage_context,service_context=service_context)
        query_engine = index.as_query_engine()
        response = query_engine.query("Select up to 3 quizzes from the document that you think I would like to take based on this input:" + user_input + ". Only choose links in the provided document. Return the quizzes as a list of URLs.")
        content = str(response)
        if "document" in content or "input" in content or "[]" in content:
            #content = "Oh no, you stumped me! I couldn't find any relevant quizzes. Perhaps try again?"
            response = query_engine.query("Select 3 random quizzes from the document and return them to me as a list of URLs.")
            content = str(response)
    except RateLimitError:
        content = "The server is experiencing a high volume of requests. Please try again later."

    return jsonify(content=content)

if __name__ == '__main__':
    app.run(debug=True)