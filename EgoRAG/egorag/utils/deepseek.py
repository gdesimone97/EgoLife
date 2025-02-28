import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential


def deepseek_chat(
    user_message,
    system_message,
    model_name="DeepSeek-R1",
    endpoint=None,
    key=None,
    max_tokens=4096,
):
    # Set environment variables or use defaults
    endpoint = endpoint or os.getenv(
        "AZURE_INFERENCE_SDK_ENDPOINT",
        "Your_Endpoint",
    )
    model_name = model_name or os.getenv("DEPLOYMENT_NAME", "DeepSeek-R1")
    key = key or os.getenv("AZURE_INFERENCE_SDK_KEY", "YOUR_KEY_HERE")

    # Initialize the client
    client = ChatCompletionsClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    max_retries = 3
    retry_count = 0
    # Prepare and send the request
    while retry_count < max_retries:
        try:
            response = client.complete(
                messages=[
                    SystemMessage(content=system_message),
                    UserMessage(content=user_message),
                ],
                model=model_name,
                max_tokens=max_tokens,
            )
            full_response = response["choices"][0]["message"]["content"]
            return full_response
        except Exception as e:
            retry_count += 1
            print(f"Error on attempt {retry_count}: {e}")
            if retry_count == max_retries:
                print("Max retries reached. Giving up.")
                return None
