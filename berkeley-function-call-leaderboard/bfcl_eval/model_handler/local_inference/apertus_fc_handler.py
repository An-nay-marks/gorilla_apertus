import json
import threading
import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import convert_to_tool, default_decode_execute_prompting


class ApertusFCHandler(BaseHandler):
    """In-process Hugging Face handler for Apertus-style function-calling models.

    - Loads model/tokenizer from `--local-model-path` when supplied, otherwise falls back to
      the `model_name` passed in the ModelConfig.
    - Expects the model to return JSON text representing function calls and/or tool outputs.

    Notes:
    - This implementation is intentionally conservative about concurrency: it serializes
      calls to the model.generate() through a lock. For higher throughput, run multiple
      processes (one per GPU) or use the OSS/server approach.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        local_model_path: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_path_or_id = local_model_path if local_model_path else model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy load to avoid heavy startup cost during import
        self._loaded = False
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    def _load_model(self):
        with self._load_lock:
            if self._loaded:
                return
            # When a local path is provided, prefer local files only to avoid network calls
            local_only = False
            try:
                # Heuristic: if model_path_or_id looks like a path on disk
                import os

                local_only = os.path.isdir(self.model_path_or_id)
            except Exception:
                local_only = False

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path_or_id,
                local_files_only=local_only,
                trust_remote_code=True,
            )

            # Model
            # We use device_map='auto' when CUDA available to let accelerate allocate weights.
            # For CPU-only machines it falls back to standard loading.
            model_load_kwargs: Dict[str, Any] = dict(trust_remote_code=True)
            if "cuda" in self.device and torch.cuda.is_available():
                model_load_kwargs.update({"device_map": "auto", "torch_dtype": torch.float16})
            else:
                model_load_kwargs.update({"device_map": None})

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path_or_id, low_cpu_mem_usage=True, local_files_only=local_only, **model_load_kwargs
            )
            self.model.eval()

            # max context
            if hasattr(self.model.config, "max_position_embeddings"):
                self.max_context_length = self.model.config.max_position_embeddings
            else:
                self.max_context_length = self.tokenizer.model_max_length

            self._loaded = True

    #### FC methods ####
    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        # For FC we store the function specs under "function" and initialize a message list
        functions = test_entry.get("function", [])
        inference_data.setdefault("function", functions)
        inference_data.setdefault("message", [])
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        # Convert provided function docs to the model-facing tool representation if needed
        functions = test_entry.get("function", [])
        # Use helper to convert to the canonical 'tool' structure used elsewhere if available
        # We simply keep the JSON schema in inference_data for prompt formatting
        inference_data["function_specs"] = functions
        return inference_data

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(self, inference_data: dict, model_response_data: dict) -> dict:
        # model_response_data contains 'model_responses' (text). Append as assistant message.
        inference_data["message"].append({"role": "assistant", "content": model_response_data["model_responses"]})
        return inference_data

    def _add_execution_results_FC(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        # Add each execution result back as 'tool' role messages. The framework expects this for next turn.
        for execution_result, decoded_model_response in zip(execution_results, model_response_data.get("model_responses_decoded", [])):
            inference_data["message"].append({"role": "tool", "content": execution_result})
        return inference_data

    def _format_prompt(self, messages, function):
        # The user provided a Jinja-like template that formats messages with special tokens.
        # Here we implement a simplified rendering: we will join messages using the tokens similar
        # to the example. For more exact behavior, you can adapt this function to run the
        # template with jinja2 and pass 'messages' and 'tools' into the template.
        parts = []
        # system message if present
        if messages and len(messages) > 0 and messages[0].get("role") == "system":
            system_content = messages[0].get("content")
            parts.append("<|system_start|>" + (system_content or "") + "<|system_end|>")
            loop_messages = messages[1:]
        else:
            parts.append("<|system_start|>You are Apertus, a helpful assistant created by the SwissAI initiative.\nKnowledge cutoff: 2024-04\nCurrent date: " + time.strftime("%Y-%m-%d") + "<|system_end|>")
            loop_messages = messages

        # developer block (we disable thinking by default)
        dev_block = "<|developer_start|>Deliberation: disabled\n"
        if function:
            # add a simple rendering of tool capabilities
            dev_block += "Tool Capabilities:\n"
            for f in function:
                name = f.get("name")
                desc = f.get("description", "")
                dev_block += f"// {desc}\ntype {name} = (_) => any;\n"
        else:
            dev_block += "Tool Capabilities: disabled\n"
        dev_block += "<|developer_end|>"
        parts.append(dev_block)

        # messages
        for message in loop_messages:
            role = message.get("role")
            if role == "user":
                parts.append("<|user_start|>" + message.get("content", "") + "<|user_end|>")
            elif role == "assistant":
                parts.append("<|assistant_start|>" + message.get("content", "") + "<|assistant_end|>")
            elif role == "tool":
                parts.append(message.get("content", ""))
            else:
                parts.append(message.get("content", ""))

        # Add assistant generation token
        parts.append("<|assistant_start|>")

        return "".join(parts)

    def _query_FC(self, inference_data: dict):
        # Ensure model loaded
        self._load_model()

        function = inference_data.get("function", [])
        message = inference_data.get("message", [])

        prompt = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}

        # Tokenize prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_token_count = input_ids.shape[1]

        # Lock generation to be safe with threading
        with self._generate_lock:
            start_time = time.time()
            # We ask the model to generate a continuation; adjust generation kwargs as needed
            input_ids = input_ids.to(self.model.device)
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=self.temperature,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            end_time = time.time()

        # Decode generated portion only
        generated_ids = outputs[0][input_ids.shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Mock an API-like response object
        api_response = {
            "text": text,
            "usage": {"prompt_tokens": int(input_token_count), "completion_tokens": int(generated_ids.shape[0])},
        }
        latency = end_time - start_time
        return api_response, latency

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        result = result.replace("<|python_tag|>", "")
        # Llama sometimes separates the function calls with `;` and sometimes with `,`
        if ";" in result:
            """
            "<|python_tag|>{\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"10\", \"k\": \"3\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"15\", \"k\": \"5\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"20\", \"k\": \"7\", \"p\": \"0\"}}"
            """
            function_calls = result.split(";")
            function_calls = [json.loads(func_call) for func_call in function_calls]
        else:
            """
            "[\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"20\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"12\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"10\", \"k\": \"3\"}}\n]"
            """
            function_calls = eval(result)
            if type(function_calls) == dict:
                function_calls = [function_calls]

        decoded_output = []
        for func_call in function_calls:
            name = func_call["name"]
            params = func_call["parameters"]
            decoded_output.append({name: params})

        return decoded_output
    
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        # The model returns text that includes tool calls and tool outputs in the Apertus template.
        # We will return the raw text as 'model_responses' and the token usage.
        return {
            "model_responses": api_response.get("text", ""),
            "input_token": api_response.get("usage", {}).get("prompt_tokens", 0),
            "output_token": api_response.get("usage", {}).get("completion_tokens", 0),
        }

    def decode_execute(self, result, has_tool_call_tag: bool):
        # The model is expected to emit JSON segments for function/tool calls and tool outputs.
        # We attempt to extract a JSON array of function calls or a single object.
        # If parsing fails, we return the raw text to allow the caller to treat as plain output.
        text = result
        # Try to locate a JSON array/object in the text
        try:
            # A simple heuristic: find first '{' or '[' and last matching '}' or ']'
            first = min([i for i in [text.find('{'), text.find('[')] if i != -1])
            candidate = text[first:]
            parsed = json.loads(candidate)
            # If parsed is a list of calls or a dict representing a call, normalize
            return parsed
        except Exception:
            # Fall back to default decoder for prompting style
            return default_decode_execute_prompting(result, has_tool_call_tag)
