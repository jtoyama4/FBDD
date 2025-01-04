import os
import sys
import time
import base64
import argparse
import requests
import pandas as pd
from tqdm import tqdm
import PIL

import openai
import anthropic
import google.generativeai as genai


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cf',
        action='store_true',
        help='Set this to True if you want to specify a counter-factual prompt.'
    )
    parser.add_argument(
        '--ct',
        action='store_true',
        help='Set this to True if you want to specify a context prompt.'
    )
    parser.add_argument(
        '--selected_id',
        type=str,
        default=None,
        help='Specify the txt file that lists IDs you want to test. '
             'If not specified, all IDs will be tested.'
    )
    return parser.parse_args()


def create_dataframe(csv_file_path, images_folder_path):
    """
    Read a CSV file and create a pandas DataFrame.
    Also add a column 'image_path' that points to each image file.
    
    :param csv_file_path: str, path to the CSV file
    :param images_folder_path: str, path to the image folder
    :return: pd.DataFrame
    """
    data = pd.read_csv(csv_file_path)
    data['image_path'] = data['ID'].apply(
        lambda id_num: os.path.join(images_folder_path, f"{id_num}.png")
    )
    return data


def get_content_from_openai(openai_api_key, question, image_path):
    """
    Send a request to the OpenAI GPT-4 model (via chat completions) with an image.
    
    :param openai_api_key: str, API key for OpenAI
    :param question: str, prompt text
    :param image_path: str, path to the image file
    :return: str, response content
    """
    # Encode image to Base64
    with open(image_path, "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{question}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    # Extract the response text
    content = response.json()['choices'][0]['message']['content']
    return content


def get_content_from_claude3(client, question, image_path):
    """
    Send a request to the Anthropic Claude 3 model with an image in base64.
    
    :param client: anthropic.Anthropic, the Anthropic API client
    :param question: str, prompt text
    :param image_path: str, path to the image file
    :return: str, response content
    """
    with open(image_path, "rb") as image_file:
        encoded_img = base64.b64encode(image_file.read()).decode('utf-8')

    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": 'user',
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_img
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    )
    content = message.content[0].text
    return content


def get_content_from_gemini(model, question, image_path):
    """
    Send a request to Google's Gemini model with an image and a prompt.
    
    :param model: genai.GenerativeModel, a configured GenerativeModel instance
    :param question: str, prompt text
    :param image_path: str, path to the image file
    :return: str, response content
    """
    img = PIL.Image.open(image_path)
    response = model.generate_content([question, img], stream=True)
    response.resolve()
    try:
        content = response.text
    except Exception:
        # When response is more complex, gather parts manually
        text_parts = []
        parts = response.candidates[0].content.parts
        for part in parts:
            text_parts.append(part.text)
        content = ''.join(text_parts)

    return content


def access_api_with_retry(api_function, api_args, max_retries=5, wait_time=60):
    """
    Retry API access if an exception is raised, up to max_retries times.
    
    :param api_function: function, the function that calls the API
    :param api_args: list, arguments for api_function in the correct order
    :param max_retries: int, maximum number of retries
    :param wait_time: int, waiting time (seconds) between retries
    :return: str, response or error message
    """
    retries = 0
    while retries < max_retries:
        try:
            response = api_function(*api_args)
            return response
        except Exception as e:
            print(f"Error occurred: {e}")
            retries += 1
            time.sleep(wait_time)

    return "Error occurred after maximum retries."


def main():
    # Parse arguments
    args = parse_args()

    # Constants
    CSV_FILE_PATH = '../dataset/dataset_meta.csv'
    IMAGES_FOLDER_PATH = '../dataset/img_dataset'

    # Create DataFrame
    data = create_dataframe(CSV_FILE_PATH, IMAGES_FOLDER_PATH)

    # Prepare result DataFrame
    result_df = pd.DataFrame(columns=[
        'ID', 'Answer GPT4', 'Answer Claude3', 'Answer GeminiPro'
    ])

    # Retrieve API keys & configure
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    google_api_key = os.environ.get('GOOGLE_API_KEY')

    # Initialize clients / models
    anthropic_client = anthropic.Anthropic()
    genai.configure(api_key=google_api_key)
    google_model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # Determine default prefix
    if args.ct:
        default_prefix = (
            "Please answer based solely on the image without relying on any "
            "existing knowledge. Think carefully."
        )
    else:
        default_prefix = "Please answer the question."

    # If --selected_id is provided, read the IDs from file
    if args.selected_id:
        with open(args.selected_id, 'r') as file:
            selected_id_list = [line.strip() for line in file if line.strip()]
        print("Selected IDs found. Following IDs will be tested:")
        print(selected_id_list)
    else:
        selected_id_list = None

    # Create folder to store the results
    timestr = time.strftime("%Y%m%d-%H%M%S")
    folder_name = f"results/experiment_results_{timestr}"
    os.makedirs(folder_name, exist_ok=True)

    file_index = 0

    # Iterate over rows in DataFrame
    for _, row in tqdm(data.iterrows(), total=len(data)):
        idn = row['ID']

        # If specific IDs are provided, skip others
        if selected_id_list and idn not in selected_id_list:
            continue

        # Build question
        question = f"{default_prefix}\n\n{row['Question']}"
        if args.cf:
            cf_prefix = row['Counterfactual prompting']
            question = f"{cf_prefix}\n\n{question}"

        image_path = row['image_path']

        # Send request to GPT-4
        content_gpt4 = access_api_with_retry(
            get_content_from_openai,
            [openai_api_key, question, image_path]
        )

        # Send request to Claude 3
        content_claude3 = access_api_with_retry(
            get_content_from_claude3,
            [anthropic_client, question, image_path]
        )

        # Send request to Gemini
        content_gemini = access_api_with_retry(
            get_content_from_gemini,
            [google_model, question, image_path]
        )

        # Append results to DataFrame
        new_row = pd.DataFrame({
            'ID': [idn],
            'Answer GPT4': [content_gpt4],
            'Answer Claude3': [content_claude3],
            'Answer GeminiPro': [content_gemini]
        })
        result_df = pd.concat([result_df, new_row], ignore_index=True)

        # Save intermediate results to Excel each iteration
        if args.cf:
            result_file_path = os.path.join(
                folder_name,
                f"result_{timestr}_counterfactual_{file_index}.xlsx"
            )
        elif args.ct:
            result_file_path = os.path.join(
                folder_name,
                f"result_{timestr}_context_{file_index}.xlsx"
            )
        else:
            result_file_path = os.path.join(
                folder_name,
                f"result_{timestr}_{file_index}.xlsx"
            )

        result_df.to_excel(result_file_path, index=False)
        file_index += 1

        # Sleep to avoid hitting rate limits (if necessary)
        time.sleep(5)


if __name__ == "__main__":
    main()
