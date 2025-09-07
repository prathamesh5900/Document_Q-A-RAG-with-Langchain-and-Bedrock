import boto3
import botocore.config
import json
from datetime import datetime


## Usecase - Blog Generation

def blog_generate_using_bedrock(blogtopic:str)-> str:
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                   You are a helpful assistant.<|eot_id|>
                   <|start_header_id|>user<|end_header_id|>
                    Write a 200 word blog on the topic {blogtopic}.<|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>"""
    


    body ={
        "prompt":prompt,
        "max_gen_len":512,
        "temperature":0.5,
        "top_p":0.9
    }

    try:
        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId="meta.llama3-8b-instruct-v1:0",  
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response.get("body").read())
        blog_details = response_body["generation"]
        print(response_body)
        return blog_details
    
    except Exception as e:
        print(f"Error generating the blog:{e}")
        return ""
    
def save_blog_details_s3(s3_key,s3_bucket,generate_blog):
    s3 = boto3.client("s3")


    try:
        s3.put_object(Bucket= s3_bucket, Key = s3_key , Body = generate_blog)
        print("Code saved to S3")

    except:
        print("Error when saving code to s3")



    

def lambda_handler(event, context):
    # TODO implement
    event = json.loads(event["body"])
    blogtopic = event["blog_topic"]


    generate_blog = blog_generate_using_bedrock(blogtopic=blogtopic)

    if generate_blog:
        current_time = datetime.now().strftime("%H%M%S")
        s3_key = f"blog-output-floder/{current_time}.txt"
        s3_bucket = "aws_bedrock_course1"
        save_blog_details_s3(s3_key,s3_bucket,generate_blog)


    else:
        print("No blog was generated")

    return {
        "statuscode":200,
        "body":json.dumps("Blog generation is completed")
    }






    

