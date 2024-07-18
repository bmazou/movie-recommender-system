import json

import requests
from openai import OpenAI
from prompt_manager import PromptManager


class LLM:
    def __init__(self, model):
        """Initializes the LLM class with the specified model, either by opening a connection to the OpenAI API or by using the Hugging Face API."""
        if model not in ["gpt-3.5", "gpt-4o", "mistral", "mixtral"]:
            raise ValueError("Invalid LLM model name")
        
        with open("config.json", "r") as f:
            config = json.load(f)
        
        self.model_name = model
        self.api_key = self.load_api_key(model, config)
        self.api_url = self.load_api_url(model, config)
        self.client = self.load_client(model)
        
    def load_api_key(self, model, config):
        if model == "gpt-3.5" or model == "gpt-4o":
            return config["openai_api_key"]
        else:
            return config["huggingface_api_key"]

        # return config[f'{model}_api_key']
        
    def load_api_url(self, model, config):
        if model == "mistral" or model == "mixtral":
            return config[f'{model}_api_url']
        else:
            return None
    
    def load_client(self, model):
        if model == "gpt-3.5" or model == "gpt-4o":
            return OpenAI(api_key = self.api_key)
        else:
            return None
    
    def generate_open_ai(self, prompt):
        """ Generates a response from the OpenAI API based on the given prompt. Uses in-context learning by providing the conversation history to the LLM.
        Args:
            prompt (str): The prompt to generate a response for. Ends with "<LINEBREAK>" to indicate when the user's or assistant's turn ends. 
        Returns:
            str: The generated response.
        """
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        conversation = prompt.strip().split("<LINEBREAK>")[:-1]
        conversation = [text.strip() for text in conversation]
        

        roles = ['user', 'assistant']  
        for i, text in enumerate(conversation):
            role = roles[i % 2]  
            messages.append({"role": role, "content": text})
        	
        
        if self.model_name == "gpt-3.5":
            model_full_name = "gpt-3.5-turbo-0125"
        else:
            model_full_name = "gpt-4o"
         
        response = self.client.chat.completions.create(
            model=model_full_name,
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def generate_huggingface(self, prompt, max_length):
        """Generates a response from the Hugging Face API based on the given prompt."""
        def get_answer_from_text(text):
            parts = text.split("[/INST]")
            return parts[-1].replace("{self.id_end}", "").strip()
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_length}
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(self.api_url, headers=headers, json=payload)
        text = response.json()[0]["generated_text"]
        return get_answer_from_text(text)
        
        
    def generate(self, prompt, max_length=2000):
        """ Routes the prompt to the appropriate API based on the model used."""
        if self.client:
            return self.generate_open_ai(prompt)
        else:
            return self.generate_huggingface(prompt, max_length)
        
    @staticmethod
    def split_llm_answer(text):
        """ Splits the LLM answer into movie ratings and general preferences.
        Args:
            text (str): The LLM answer containing the movie ratings and general preferences.
        Returns:
            tuple: A tuple containing the movie ratings and general preferences. Or None if the input is invalid.
        """
    
        try:
            movie_part, general_part = text.split('---')
        except ValueError:
            return None, None


        movie_part = movie_part.strip()
        # if general_part contains <EMPTY> then return None
        general_part = general_part.strip() if not "<EMPTY>" in general_part else None
        if "<EMPTY>" in movie_part:
            return None, general_part

        try:
            movie_part = movie_part.split(';;;')
            movie_ratings = [(movie.rsplit(': ', 1)[0], float(movie.rsplit(': ', 1)[1])) for movie in movie_part]
        except (ValueError, IndexError):
            return None, general_part

        return movie_ratings, general_part