import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords


nltk.download("stopwords")
nltk.download('punkt')
stop_words = set(stopwords.words("english"))


def delete_text_within_parentheses(text):
    # Define regex pattern to match parentheses and any text within them
    pattern = r"\([^)]*\)"

    # Use regex to find all matches of the pattern in the input text
    matches = re.findall(pattern, text)

    # Iterate through the matches and replace them with empty strings
    for match in matches:
        text = text.replace(match, "")

    return text


def remove_extra_spaces(text):
    return " ".join(text.split())


def clean_text(text):
    # Remove all characters except letters, numbers, and spaces
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return cleaned_text


def remove_newlines(text):
    return text.replace("\n", "")


def remove_key_points(text):
    return text.replace("Key Points\n                    - ", "")


def parse_xml_folder(xml_folder_path, json_file_path):
    df_list = []
    for filename in tqdm(os.listdir(xml_folder_path)):
        if filename.endswith(".xml"):
            xml_file_path = os.path.join(xml_folder_path, filename)
            with open(xml_file_path, "r") as f:
                data = f.read()

            bs_data = BeautifulSoup(data, "xml")
            b_unique = bs_data.find_all("Question")

            questions = []
            for i in b_unique:
                question = i.get_text("qty")
                question = delete_text_within_parentheses(question)
                questions.append(question)

            b_unique = bs_data.find_all("Answer")
            answers = []
            for i in b_unique:
                answer = i.get_text("qty")
                answer = remove_key_points(answer)
                answer = answer.replace("\n", "")
                answer = answer.split()
                answer = " ".join(answer)
                answer = delete_text_within_parentheses(answer)
                answers.append(answer)

            document_tag = bs_data.find("Document")
            url = document_tag["url"]

            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            text = ""

            for tag in soup.find_all(["p", "ul"]):
                if tag.name == "p":
                    if len(tag.text) < 300 or "cookies" in tag.text.lower():
                        continue
                    else:
                        text += tag.text
                elif tag.name == "ul":
                    for em_tag in tag.find_all("li"):
                        if len(em_tag.text) < 100:
                            continue
                        else:
                            text += em_tag.text

            text = text.replace("\n", "")
            text = text.split()
            text = " ".join(text)
            tokens = nltk.word_tokenize(text)
            filtered_tokens = [word for word in tokens if not word in stop_words]
            filtered_text = " ".join(filtered_tokens)
            filtered_text = delete_text_within_parentheses(filtered_text)

            df = pd.DataFrame([questions, answers]).T
            # df.columns = ['Questions', 'Answers']
            # df['URL_text'] = text

            qa_list = []
            for index, row in df.iterrows():
                qa_dict = {"Question": row[0], "Answer": row[1]}
                qa_list.append(qa_dict)

            df_list.append({"Context": filtered_text, "QA": qa_list})

    with open(json_file_path, "w") as f:
        json.dump(df_list, f)


parse_xml_folder("MedQuAD/9_CDC_QA", "JSON_files/test.json")
parse_xml_folder("MedQuAD/3_GHR_QA", "JSON_files/train.json")
