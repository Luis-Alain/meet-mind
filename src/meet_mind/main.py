import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def use_converse_api():
    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
    model_id = os.environ.get("AWS_BEDROCK_MODEL_ID")

    system_prompt = "You are a helpful assistant that provides concise and accurate answers to user questions about AI Agents."

    user_message = "What is an AI Agent?"

    print(f"Using Bedrock Converse API with model: {model_id}")
    print("=" * 60)
    print(f"System Prompt: {system_prompt}")
    print(f"User Message: {user_message}\n")

    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            system=[{"text": system_prompt}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"temperature": 0.7, "maxTokens": 2000},
        )

        # Extract the response
        output_text = response["output"]["message"]["content"][0]["text"]
        print("Response from Bedrock Converse API:")
        print("=" * 60)
        print(output_text)

        # Display token usage
        usage = response.get("usage", {})
        print("\nToken Usage:")
        print(f"Input Tokens: {usage.get('inputTokens', 'N/A')}")
        print(f"Output Tokens: {usage.get('outputTokens', 'N/A')}")
        print(f"Total Tokens: {usage.get('totalTokens', 'N/A')}")

        print(f"\nStop Reason: {response.get('stopReason')}")

        return response

    except Exception as e:
        print(f"Error using Converse API: {e}")
        raise


def main() -> None:
    use_converse_api()


if __name__ == "__main__":
    main()