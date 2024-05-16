from flask import Flask, render_template, request
from pyngrok import ngrok
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



#loading the model
model_id = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Define the conversation context
document = """
mefic docs
"""
port = "5000"



def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant called lido. The assistant gives helpful, detailed answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + conversation

    return formatted_input
def get_response(messages):
  formatted_input = get_formatted_input(messages, document)
  tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

  terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

  response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
  print(tokenizer.decode(response, skip_special_tokens=True))
  return response


# Start flask app and set to ngrok
app = Flask(__name__)
# Open a ngrok tunnel to the HTTP server
public_url = ngrok.connect(port).public_url

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an response of {prompt}")
  messages = [
    {"role": "user", "content": prompt}
  ]
  res = get_response(messages)
  print("Sending image ...")
  return render_template('index.html', response=res)


if __name__ == '__main__':
    app.run()