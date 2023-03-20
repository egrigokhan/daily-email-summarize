import imaplib
import email
import smtplib
from email.mime.text import MIMEText
from datetime import date, timedelta
import time
import re
from html import unescape

# Define a function to remove HTML and CSS


def remove_html_css(text):
    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text)
    # Remove CSS styles
    text = re.sub('<style[^<]+?</style>', '', text, flags=re.DOTALL)
    # Decode any HTML entities
    text = unescape(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\S+', '', text)
    # Remove any remaining whitespace
    text = text.strip()
    return text


def get_daily_email_summary(email_address, email_password, imap_server, smtp_server, smtp_port, smtp_username, smtp_password):
    # IMAP server settings
    IMAP_SERVER = imap_server
    IMAP_PORT = 993
    IMAP_USERNAME = email_address
    IMAP_PASSWORD = email_password

    # SMTP server settings
    SMTP_SERVER = smtp_server
    SMTP_PORT = smtp_port
    SMTP_USERNAME = smtp_username
    SMTP_PASSWORD = smtp_password

    # Connect to the IMAP server and select the inbox
    imap_server = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
    imap_server.login(IMAP_USERNAME, IMAP_PASSWORD)
    imap_server.select('inbox')

    # Get the date of today
    today = date.today()

    # Set the search criteria for emails received today
    search_criteria = f'(SINCE "{today.strftime("%d-%b-%Y")}")'

    # Search for emails from the specified email address received today
    status, messages = imap_server.search(
        None, search_criteria)

    # Get the list of message IDs as a list of strings
    message_ids = messages[0].split(b' ')

    messages_flt = []

    # Loop through the message IDs and retrieve each message
    for message_id in message_ids:
        status, msg = imap_server.fetch(message_id, '(RFC822)')
        email_message = email.message_from_bytes(msg[0][1])
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    body = part.get_payload(decode=True)
                    body = remove_html_css(body.decode())
                    messages_flt.append({
                        "from": email_message["From"],
                        "body": body
                    })
                    # print(f"Message {message_id.decode()}: {body}")

    print(f"Found {len(messages_flt)} messages")

    # Close the IMAP connection
    imap_server.close()
    imap_server.logout()

    # return
    return messages_flt


def format_emails_into_prompt(emails):
    prompt = ""
    for email in emails:
        prompt += f"{email['from']}: {email['body']}\n\n"
        prompt += "*** *** ***\n\n"

    return prompt
