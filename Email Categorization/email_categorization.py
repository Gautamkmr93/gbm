#Author: @Gautam Kumar

import os
import json
import re
import logging
from typing import Optional,Dict,List



logging.basicConfig(filename="email_categorization.log",filemode="a",format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def clean_body(raw_body: str) -> Optional[str]:
    if not raw_body or not isinstance(raw_body, str):
        logging.warning("Invalid input provided or empty string has been given.")
        return None
    try:
        body = raw_body
        logging.info("cleanning the email body.")
        body = re.sub(r"(?i)(X-[\w-]+|Mime-Version|Content-Type|Content-Transfer-Encoding|X-Folder|X-Origin|X-FileName):.*",
            "", body
        )
        body = re.split(r"(?i)-----Original Message-----", body)[0]
        body = re.split(r"(?i)From: .*@.*", body)[0]
        body = re.sub(r"(?i)(To|From|Subject|Cc|Bcc):.*", "", body)
        body = re.sub(r"http[s]?://\S+", "", body)
        body = re.sub(r"(?i)Get your FREE.*", "", body)
        body = re.sub(r"\n\s*\n+", "\n", body)
        body = re.sub(r"\s{2,}", " ", body)
        cleaned_body = body.strip()
        logging.info("Email body cleaned successfully.")
        return cleaned_body
    except Exception as e:
        logging.exception("Error has been occured while cleaning email body: %s", str(e))
        return None


def extract_email_data(file_path: str) -> Optional[Dict[str, str]]:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        sender_match = re.search(r"From:\s*(.*)", content)
        subject_match = re.search(r"Subject:\s*(.*)", content)
        body_start = subject_match.end() if subject_match else 0
        raw_body = content[body_start:].strip()
        cleaned_body = clean_body(raw_body)
        extracted_data = {
            "from": sender_match.group(1).strip() if sender_match else "",
            "subject": subject_match.group(1).strip() if subject_match else "",
            "body": cleaned_body if cleaned_body else ""
        }
        logging.info("Successfully extracted email data from: %s", file_path)
        return extracted_data
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
    except Exception as e:
        logging.exception("exception while extracting email data from %s: %s", file_path, str(e))
    return None


def process_folder(folder_path: str, output_json: str) -> None:
    if not os.path.isdir(folder_path):
        logging.error("Invalid folder path: %s", folder_path)
        return
    extracted_data: List[Dict[str, str]] = []
    files_processed = 0
    files_skipped = 0

    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if not os.path.isfile(file_path):
                logging.warning("Skipping non-file: %s", file_path)
                continue

            try:
                email_data: Optional[Dict[str, str]] = extract_email_data(file_path)
                if email_data:
                    extracted_data.append(email_data)
                    files_processed += 1
                else:
                    files_skipped += 1
                    logging.warning("Extraction failed or returned empty for: %s", filename)
            except Exception as e:
                logging.exception("Error processing file %s: %s", filename, str(e))
                files_skipped += 1
        with open(output_json, 'w', encoding='utf-8') as json_file:
            json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)

        logging.info(
            "Processing complete. %d files processed, %d files skipped. Output saved to %s",
            files_processed, files_skipped, output_json
        )
    except Exception as e:
        logging.exception("Critical failure during folder processing: %s", str(e))


def classify_email(subject: Optional[str], body: Optional[str]) -> str:
    try:
        if not subject:
            subject = ""
        if not body:
            body = ""
        combined = f"{subject.lower()} {body.lower()}"
        for category, keywords in CATEGORIES.items():
            for keyword in keywords:
                if keyword.lower() in combined:
                    logging.info("Email classified as '%s' using keyword '%s'", category, keyword)
                    return category
        logging.info("Email classified as others'")
        return "others"
    except Exception as e:
        logging.exception("Error during email classification: %s", str(e))
        return "others"



if __name__ == "__main__":
    folder_path = "/Users/gautamkumar/Desktop/GBM/Email Categorization/Emails"
    output_json = "Intermediate.json"
    process_folder(folder_path, output_json)

    with open('Intermediate.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        data_ = data.pop(0)

    CATEGORIES = {
    "Market/Finance": ["chart", "matrix", "api", "crude", "stocks", "transactions", "amount", "finance"],
    "HR/Admin": ["calendar", "ranked", "play", "camp", "paperwork", "update", "Y", "schedule"],
    "Personal": ["love", "hello", "thankful", "sorry", "greg", "personal"],
    "Technical": ["delivery point", "website", "server", "ECA", "records", "data"],}

    processed_data = []
    for email in data:
        try:
            sender = email.get("from", "")
            subject = email.get("subject", "")
            body = email.get("body", "")
            category = classify_email(subject, body)
            processed_email = {
                "sender": sender,
                "subject": subject,
                "body": body,
                "category": category
            }
            processed_data.append(processed_email)
        except Exception as e:
            logging.error(f"Failed to process email: {e}")

    try:
        with open("output_file.json", "w") as f:
            json.dump(processed_data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to export JSON: {e}")
