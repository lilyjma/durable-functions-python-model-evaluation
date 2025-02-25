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
    user_prompt = json.loads(req.get_body().decode())
    print(f"User Prompt: {user_prompt}")
    
    instance_id = await client.start_new("orchestrator_function", client_input=user_prompt)
    response = client.create_check_status_response(req, instance_id)
    return response

@app.orchestration_trigger(context_name="context")
def orchestrator_function(context):
    """
    Orchestrator function to call models in parallel and evaluate the best response.
    """
    user_prompt = context.get_input()
    
    if not user_prompt:
        raise ValueError("Please provide a question to ask the models.")
    
    retry_interval_in_milliseconds = 2000
    max_number_of_attempts = 3
    retry_options = df.RetryOptions(retry_interval_in_milliseconds, max_number_of_attempts)
    
    tasks = [
        context.call_activity_with_retry("get_gpt35_result", retry_options, user_prompt),
        context.call_activity_with_retry("get_gpt4omini_result", retry_options, user_prompt),
        context.call_activity_with_retry("get_phi4_result", retry_options, user_prompt)
    ]
    
    results = yield context.task_all(tasks)
    
    proposed_responses = "\n".join(f"{i+1}. {element}" for i, element in enumerate(results))

    final_answer = yield context.call_activity_with_retry("get_gpt4_result", retry_options, [proposed_responses, user_prompt])
    return final_answer

@app.activity_trigger(input_name="prompt")
def get_gpt35_result(prompt: str):
    system_prompt = """You are a helpful assistant in STEM."""
    
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    response = client.complete(
        model="gpt-35-turbo", # model deployment name
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=prompt)
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

@app.activity_trigger(input_name="prompt")
def get_gpt4omini_result(prompt: str):
    system_prompt = """You are a helpful assistant in STEM."""
    
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    response = client.complete(
        model="gpt-4o-mini", # model deployment name
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=prompt)
        ],
        temperature=0
    )

    return response.choices[0].message.content

@app.activity_trigger(input_name="prompt")
def get_phi4_result(prompt: str):
    system_prompt = """You are a helpful assistant in STEM."""
    
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    response = client.complete(
        model="Phi-4", # model deployment name
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=prompt)
        ],
        temperature=0
    )

    return response.choices[0].message.content

@app.activity_trigger(input_name="inputData")
def get_gpt4_result(inputData: list):
    system_prompt = """
    You have been provided with a set of responses from various language models to a user query. 
    Your task is to pick out the response that's the simplest and most intuitive. 

    Responses from models:"""
    
    client = ChatCompletionsClient(
        endpoint=os.environ["MODEL_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["MODEL_API_KEY"]),
    )
    
    proposed_responses, user_prompt = inputData[0], inputData[1]
    complete_system_prompt = system_prompt + "\n" + proposed_responses
    
    print(f"Evaluation model input: \n {complete_system_prompt}")
    
    response = client.complete(
        model="gpt-4", # model deployment name
        messages=[
            SystemMessage(content=complete_system_prompt),
            UserMessage(content=user_prompt)
        ],
        temperature=0
    )

    return response.choices[0].message.content


