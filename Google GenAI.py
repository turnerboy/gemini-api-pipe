"""
title: gemini-api-pipe
author: turnerboy
version: 0.1.4
intstructions: Create a funcition in your OpenWebUI container and paste this code and add your Gemini API key securely :)

"""

import os
import json
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from typing import List, Union, Iterator

# Toggle detailed logging
DEBUG = False


class Pipe:
    # Configuration valves for API key and safety mode
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="")        # your Google API key
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)  # whether to loosen safety filters

    def __init__(self):
        self.id = "google_genai"                        # pipe identifier
        self.type = "manifold"                          # manifold integration type
        self.name = "Google: "                          # display name prefix
        # load valves from environment
        self.valves = self.Valves(
            **{
                "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )

    def get_google_models(self):
        # return error if API key missing
        if not self.valves.GOOGLE_API_KEY:
            return [{
                "id": "error",
                "name": "GOOGLE_API_KEY is not set. Please update the API Key in the valves.",
            }]
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)  # init SDK
            models = genai.list_models()                       # fetch model list
            # filter to those that support generateContent
            return [
                {
                    "id": model.name[7:],    # strip "models/" prefix
                    "name": model.display_name,
                }
                for model in models
                if "generateContent" in model.supported_generation_methods
                if model.name.startswith("models/")
            ]
        except Exception as e:
            if DEBUG:
                print(f"Error fetching Google models: {e}")
            return [{"id": "error", "name": f"Could not fetch models: {e}"}]

    def pipes(self) -> List[dict]:
        # manifold hook to list available pipes
        return self.get_google_models()

    def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        # ensure API key is set
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"
        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)  # init SDK
            model_id = body["model"]

            # normalize model identifier
            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")
            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            messages = body["messages"]
            stream = body.get("stream", False)

            if DEBUG:
                print("Incoming body:", body)

            # extract optional system message
            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"),
                None
            )

            # build contents list for SDK
            contents = []
            for message in messages:
                if message["role"] == "system":
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part["type"] == "text":
                            parts.append({"text": part["text"]})
                        elif part["type"] == "image_url":
                            url = part["image_url"]["url"]
                            if url.startswith("data:image"):
                                data = url.split(",")[1]
                                parts.append({
                                    "inline_data": {"mime_type": "image/jpeg", "data": data}
                                })
                            else:
                                parts.append({"image_url": url})
                    contents.append({"role": message["role"], "parts": parts})
                else:
                    contents.append({
                        "role": "user" if message["role"] == "user" else "model",
                        "parts": [{"text": content}],
                    })

            # prepend system instruction if provided
            if system_message:
                contents.insert(0, {
                    "role": "user",
                    "parts": [{"text": f"System: {system_message}"}]
                })

            # choose model instantiation based on version
            if "gemini-1.5" in model_id:
                model = genai.GenerativeModel(
                    model_name=model_id, system_instruction=system_message
                )
            else:
                model = genai.GenerativeModel(model_name=model_id)

            # set up generation parameters
            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            # decide safety settings
            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            if DEBUG:
                print("Google API request:", model_id, contents, generation_config, safety_settings, stream)

            # handle streaming versus single response
            if stream:
                def stream_generator():
                    response = model.generate_content(
                        contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        stream=True,
                    )
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                return stream_generator()
            else:
                response = model.generate_content(
                    contents,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False,
                )
                return response.text

        except Exception as e:
            if DEBUG:
                print(f"Error in pipe method: {e}")
            return f"Error: {e}"
