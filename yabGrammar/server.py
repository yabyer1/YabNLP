from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Flask(__name__)
CORS(app)
MODEL_NAME = "TheBloke/Mistral-7B-GGUF"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
@app.route("/autocomplete", methods=["POST"])
def autocomplete():
    data = request.json
    user_text = data.get("text", "").strip()
    if not user_text:
        return jsonify({"completion": ""})
    prompt = f"Complete this sentence: {user_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=50)

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({"completion": completion[len(prompt):].strip()})
if __name__ == "__main__":
    app.run(port=5000, debug=True)