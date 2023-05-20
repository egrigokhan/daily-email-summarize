import os
import pathlib
import pickle
import subprocess
import tempfile

import requests
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
# -*- coding: utf-8 -*-
from langchain.llms import OpenAI, OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from src.lib.email import format_emails_into_prompt, get_daily_email_summary


def print_answer(question):
    #  load the email data
    emails = get_daily_email_summary(email_address=os.environ.get("email_address"),
                                     email_password=os.environ.get(
                                         "email_password"),
                                     imap_server="imap.gmail.com",
                                     smtp_server="smtp.gmail.com",
                                     smtp_port=587,
                                     smtp_username=os.environ.get(
                                         "email_address"),
                                     smtp_password=os.environ.get("email_password"))

    docs = [Document(page_content=format_emails_into_prompt([e]))
            for e in emails]

    map_prompt_template = """The following is an email:


    {text}

    Write a concise summary of the email. Make sure to include the sender and the subject of the email in the summary. And make sure to let the user know if the email requires a response, like somebody saying "Sign this document" or "I'll be late to the meeting".
    SUMMARY IN BULLET POINTS:"""
    reduce_prompt_template = """The following are the summaries of a user's emails:


    {text}

    Write a concise summary of all the emails, separated by a ✉️ and a new line. Make sure ALL EMAILS ARE INCLUDED IN THE SUMMARY. Make sure to include the sender and the subject of the email in the summary. And make sure to let the user know if the email requires a response, like somebody saying "Sign this document" or "I'll be late to the meeting"."""
    MAP_PROMPT = PromptTemplate(
        template=map_prompt_template, input_variables=["text"])
    REDUCE_PROMPT = PromptTemplate(
        template=reduce_prompt_template, input_variables=["text"])
    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce",
                                 return_intermediate_steps=True, map_prompt=MAP_PROMPT, combine_prompt=REDUCE_PROMPT)
    result = chain({"input_documents": docs}, return_only_outputs=False)
    return result["output_text"]

def approve(type, request):
    return "Error approving."

def run(message, history):
    answer = print_answer(message)
    return answer


def setup(config):
    os.environ['email_address'] = config["email_address"]
    os.environ['email_password'] = config["email_password"]
    os.environ['OPENAI_API_KEY'] = config["OPENAI_API_KEY"]
