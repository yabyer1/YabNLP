from flask import Flask, request, jsonify
from transformers import pipeline # type: ignore
app = Flask(__name__)

text_generator = pipeline("text-generation", model="gpt2")
@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    data = request.json
    prompt = data.get("text", "")
    if not prompt.strip():
        return jsonify({"completion": ""})
    
    result = text_generator(prompt, max_length= len(prompt.split() ) + 5)
    completion = result[0]["generated_text"][len(prompt):]
    return jsonify({"completion": completion.strip()})
if __name__ == "__main__":
    app.run(port=5000, debug=True)