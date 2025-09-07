import boto3   # AWS SDK for Python (lets you call Bedrock models).
import json


prompt_data = """
Act as a Shakespeare and write a poem on Machine Learning 
"""

bedrock = boto3.client(service_name="bedrock-runtime")   ## This is the object use to send requests to Bedrock.

payload = {
    "prompt": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt_data}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "max_gen_len": 512,   
    "temperature": 0.5,   ## temperature = 0.5 → Controls creativity (0 = deterministic, 1 = more random).
    "top_p": 0.9      ## top_p = 0.9 → Nucleus sampling (limits randomness by focusing on top probability mass).
}


body = json.dumps(payload)  ## Input JSON
model_id = "meta.llama3-70b-instruct-v1:0"  ## Model
response = bedrock.invoke_model(
    body = body,
    modelId = model_id,
    accept = "application/json",  ## Output format
    contentType = "application/json"  # Input format 
)



response_body = json.loads(response.get("body").read())   ## raw json
response_text = response_body["generation"]     ## extracts generated text (model's answer)
print(response_text)  ## Output Shakespeares poem