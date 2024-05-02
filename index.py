from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from flask_cors import CORS

import os
import csv
import json

app = Flask(__name__, template_folder="templates", static_folder='static')

# Set Hugging Face API token (replace with your own token)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Place your Hugging Face token here

class CSVLoader:
    def __init__(self, filename):
        self.filename = filename

    def load(self):
        documents = []
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row:
                        documents.append(row[3])
        except FileNotFoundError:
            print(f"Error: File '{self.filename}' not found.")
        except UnicodeDecodeError:
            print(f"Error: Unable to decode file '{self.filename}'. Please ensure correct encoding (e.g., UTF-8).")
        return documents

# Load documents and perform processing
loader = CSVLoader('responses.csv')

documents = loader.load()

class DocumentWrapper:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=20)
wrapped_documents = [DocumentWrapper(doc) for doc in documents]
texts = text_splitter.split_documents(wrapped_documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 10})
docs = retriever.get_relevant_documents("restaurant?")

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
chain = load_qa_chain(llm, chain_type="stuff")

# Define get_answer function before Flask routes
def get_answer(our_query, docs):
    relevant_docs = docs
    response = chain.invoke(input={'question': our_query, 'input_documents': docs})
    return response['output_text']

doc2 = None  # Define doc2 globally
@app.route('/')
def home():
    global doc2  # Access the global doc2 variable
    output_text = None
    if doc2:  # Check if doc2 has been set
        output_text = get_answer(doc2, docs)
    return render_template('index.html', output_text=output_text)

@app.route('/store_selected_question', methods=['POST'])
def store_selected_question():
    global doc2  # Define doc2 as a global variable
    selected_question_text = request.json.get('selectedQuestionText')
    doc2 = selected_question_text  # Assign the selected question text to doc2
    # Get the answer based on doc2
    output_text = get_answer(doc2, docs)
    return jsonify({'status': 'success', 'message': 'Selected question text stored in doc2.', 'output_text': output_text})

@app.route('/index1.html')
def index1():
    # Route for serving index1.html
    return render_template('index1.html')

@app.route('/update.html')
def update():
    # Route for serving update.html
    return render_template('update.html')

@app.route('/Home.html')
def Home():
    # Route for serving update.html
    return render_template('Home.html')
@app.route('/test.html')
def test():
    # Route for serving update.html
    return render_template('test.html')

@app.route('/add_question', methods=['POST'])
def add_question():
    try:
        data = request.json
        new_question = data.get('question')
        if new_question:
            # Read existing questions from JSON file
            with open('static/questions.json', 'r') as file:
                questions = json.load(file)
            # Add the new question
            questions.append({'header': f'Question {len(questions) + 1}', 'text': new_question})
            # Write updated questions back to JSON file
            with open('static/questions.json', 'w') as file:
                json.dump(questions, file, indent=4)
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid data.'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/remove_question', methods=['POST'])
def remove_question():
    try:
        data = request.json
        index = data.get('index')
        if index is not None:
            # Read existing questions from JSON file
            with open('static/questions.json', 'r') as file:
                questions = json.load(file)
            # Remove the question at the specified index
            if 0 <= index < len(questions):
                del questions[index]
                # Update index numbers of remaining questions
                for i in range(index, len(questions)):
                    questions[i]['header'] = f'Question {i + 1}'
                # Write updated questions back to JSON file
                with open('static/questions.json', 'w') as file:
                    json.dump(questions, file, indent=4)
                return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Invalid index.'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500



    
CORS(app)  # Enable CORS for all routes
csv_file_path = 'responses.csv'

def initialize_csv():
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = ['Date', 'Time', 'Question - 1', 'Question - 2', 'Question - 3', 'Question - 4']
            writer.writerow(headers)

def write_responses_to_csv(responses):
    date = datetime.now().strftime('%Y-%m-%d')
    time = datetime.now().strftime('%H:%M:%S')

    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for response in responses:
            row = [date, time, response['answer'], response['comments']]
            writer.writerow(row)

@app.route('/submitResponses', methods=['POST'])
def submit_responses():
    responses = request.json

    # Log responses to terminal
    print('Received responses:', responses)

    # Write responses to CSV file
    write_responses_to_csv(responses)

    # Send a response back to the client
    return jsonify({'message': 'Responses received by the server'})


if __name__ == '__main__':
    app.run(debug=True)
