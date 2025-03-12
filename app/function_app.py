from datetime import datetime
import os 
import json
import azure.functions as func
import azure.durable_functions as df
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

app = df.DFApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="orchestrators/orchestrator_function")
@app.durable_client_input(client_name="client")
async def http_start(req: func.HttpRequest, client):
    """
    HTTP trigger to start the orchestration.
    """
    
    # Get user prompt
    user_prompt = json.loads(req.get_body().decode())
    
    # Start the orchestration
    instance_id = await client.start_new("orchestrator_function", client_input=user_prompt)
    response = client.create_check_status_response(req, instance_id)
    
    return response

@app.orchestration_trigger(context_name="context")
def orchestrator_function(context):
    """
    Orchestrator function to call models in parallel and evaluate responses.
    """
    system_prompt = """You are a helpful assistant in STEM."""
    user_prompt = context.get_input()
    
    if not user_prompt:
        raise ValueError("Please provide a question to ask the models.")
    
    yield context.call_activity("log", f"PROMT: {user_prompt}\n")

    retry_interval_in_milliseconds = 2000
    max_number_of_attempts = 3
    retry_options = df.RetryOptions(retry_interval_in_milliseconds, max_number_of_attempts)
    
    tasks = [
        context.call_activity_with_retry("get_gpt35_result", retry_options, [user_prompt, system_prompt]),
        context.call_activity_with_retry("get_gpt4omini_result", retry_options, [user_prompt, system_prompt]),
        context.call_activity_with_retry("get_phi4_result", retry_options, [user_prompt, system_prompt])
    ]
    
    # Run all tasks in parallel
    results = yield context.task_all(tasks)
    
    # Log responses from modes
    for result in results:
        message = f"MODEL: {result[1]}\nTIMESTAMP: {result[2]}\nRESPONSE: {result[0]}\n"
        yield context.call_activity("log", message)
    
    # Prepare input for evaluation model
    model_responses = "\n".join(f"Response #{i+1}. {result}" for i, result in enumerate(results))
    
    # Evaluate responses 
    final_result = yield context.call_activity_with_retry("get_gpt4_result", retry_options, [model_responses, user_prompt])
    
    # Log evaluation result
    message = f"EVALUATION MODEL: {final_result[1]}\nTIMESTAMP: {final_result[2]}\n=== EVALUATION RESULT === \n{final_result[0]}\n"
    yield context.call_activity("log", message)

    return "Finished evaluation"

@app.activity_trigger(input_name="logMessage")
def log(logMessage: str):
    # Idempotency check - activity functions could run multiple times 
    if os.path.exists("mylog.log"):
        with open("mylog.log", "r") as f:
            if logMessage in f.read():
                return
    
    # Check if file "mylog.log" exists
    if os.path.exists("mylog.log"):
        with open("mylog.log", "a") as f:
            f.write(logMessage + "\n")
    else:
        with open("mylog.log", "w") as f:
            f.write(logMessage + "\n")
    return

@app.activity_trigger(input_name="prompts")
def get_gpt35_result(prompts: list):
    user_prompt, system_prompt = prompts[0], prompts[1]
    
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    response = client.complete(
        model="gpt-35-turbo", # model deployment name
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ],
        temperature=0
    )
    
    return [response.choices[0].message.content, "gpt-35-turbo", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

@app.activity_trigger(input_name="prompts")
def get_gpt4omini_result(prompts: list):  
    user_prompt, system_prompt = prompts[0], prompts[1]
 
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    response = client.complete(
        model="gpt-4o-mini", # model deployment name
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ],
        temperature=0
    )

    return [response.choices[0].message.content, "gpt-4omini", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

@app.activity_trigger(input_name="prompts")
def get_phi4_result(prompts: list): 
    user_prompt, system_prompt = prompts[0], prompts[1]
       
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    response = client.complete(
        model="Phi-4", # model deployment name
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt)
        ],
        temperature=0
    )

    return [response.choices[0].message.content, "phi4", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

@app.activity_trigger(input_name="inputData")
def get_gpt4_result(inputData: list):
    system_prompt = """
    You have been provided with a set of responses from various language models to a user query. 
    Your task is to use the specified rubric to score the responses and return scores in a specific format.   
    
    The rubric is as follows: 
    - Clarity: The response is easy to understand and free of ambiguity. 
    - Conciseness: The response is brief and to the point, without unnecessary information. 
    - Relevance: The response is directly related to the user query and provides useful information. 
    - Creativity: The response demonstrates original thinking and provides unique insights.
    
    Format your response as follows:
    Model Name: overall score written as a fraction, out of the max total score. For example, 15/20.
    - Clarity: Rank from 1 to 5, where 1 is very unclear and 5 is very clear. Written as a fraction, like 1/5. Provide a brief explanation.
    - Conciseness: Rank from 1 to 5, where 1 is very verbose and 5 is very concise. Written as a fraction, like 1/5. Provide a brief explanation.
    - Relevance: Rank from 1 to 5, where 1 is not relevant and 5 is very relevant. Written as a fraction, like 1/5. Provide a brief explanation.
    - Creativity: Rank from 1 to 5, where 1 is not creative and 5 is very creative. Written as a fraction, like 1/5. Provide a brief explanation.

    Responses from models:"""
    
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    
    model_responses, user_prompt = inputData[0], inputData[1]
    complete_system_prompt = system_prompt + "\n" + model_responses
        
    response = client.complete(
        model="gpt-4", # model deployment name
        messages=[
            SystemMessage(content=complete_system_prompt),
            UserMessage(content=user_prompt)
        ],
        temperature=0
    )
    
    return [response.choices[0].message.content, "gpt-4", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]

