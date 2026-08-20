import os
import dspy
import time

# API clients
from together import Together
from openai import OpenAI
import google.generativeai as genai
import anthropic


# Define API key access
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")  # for Local Deployment
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
PERCEPTA_API_KEY = os.environ.get("PERCEPTA_API_KEY")
MOONSHOT_API_KEY = os.environ.get("MOONSHOT_API_KEY")
KIMI_TIMEOUT_SECONDS = float(os.environ.get("KIMI_TIMEOUT_SECONDS", "900"))

TOKEN_USAGE_LOG = []
TOKEN_USAGE_CSV_PATH = os.environ.get("TOKEN_USAGE_CSV_PATH", "sonnet5_token_usage.csv")
CURRENT_KERNEL_NAME = None

# Anthropic models that reject manual budget_tokens and non-default sampling params.
_ADAPTIVE_THINKING_ANTHROPIC_PREFIXES = (
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-fable-5",
    "claude-mythos-5",
)


def _uses_adaptive_thinking(model_name: str) -> bool:
    return any(model_name.startswith(prefix) for prefix in _ADAPTIVE_THINKING_ANTHROPIC_PREFIXES)


def _anthropic_text_outputs(response) -> list[str]:
    texts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    if texts:
        return texts
    # Fallback for older SDK content shapes.
    return [
        block.text
        for block in response.content
        if hasattr(block, "text") and getattr(block, "type", None) != "thinking"
    ]


def _anthropic_create_message(client, **request_kwargs):
    """
    Create an Anthropic message, streaming when needed.

    The Anthropic Python SDK requires streaming for requests that may exceed
    ~10 minutes (common with adaptive thinking + high effort).
    """
    with client.messages.stream(**request_kwargs) as stream:
        return stream.get_final_message()

def set_current_kernel_name(name: str):
    global CURRENT_KERNEL_NAME
    CURRENT_KERNEL_NAME = name

def extract_usage(response, server_type: str, model: str) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        usage = getattr(response, "usage_metadata", None)

    if usage is None:
        return {
            "server_type": server_type,
            "model": model,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    if isinstance(usage, dict):
        input_tokens = usage.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_token_count")

        output_tokens = usage.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("candidates_token_count")

        total_tokens = usage.get("total_tokens")
        if total_tokens is None:
            total_tokens = usage.get("total_token_count")
    else:
        input_tokens = getattr(usage, "prompt_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_token_count", None)

        output_tokens = getattr(usage, "completion_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "candidates_token_count", None)

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = getattr(usage, "total_token_count", None)

    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    return {
        "server_type": server_type,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def log_usage(response, server_type: str, model: str):
    import csv
    from pathlib import Path

    usage_row = extract_usage(response, server_type, model)
    usage_row["kernel"] = CURRENT_KERNEL_NAME
    TOKEN_USAGE_LOG.append(usage_row)

    path = Path(TOKEN_USAGE_CSV_PATH)
    write_header = not path.exists() or path.stat().st_size == 0

    fields = [
        "kernel",
        "server_type",
        "model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        writer.writerow(usage_row)

    print(
        f"[TOKENS] {server_type}/{model}: "
        f"input={usage_row['input_tokens']} "
        f"output={usage_row['output_tokens']} "
        f"total={usage_row['total_tokens']} "
        f"saved_to={path}"
    )


def save_token_usage_csv(path="kimi_token_usage.csv"):
    # Kept for compatibility, but log_usage() now writes immediately.
    print(
        f"[TOKENS] token usage is already being appended to {TOKEN_USAGE_CSV_PATH}"
    )
def query_server(
    prompt: str | list[dict],  # string if normal prompt, list of dicts if chat prompt,
    system_prompt: str = "You are a helpful assistant",  # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0, # nucleus sampling
    top_k: int = 50, 
    max_tokens: int = 128,  # max output tokens to generate
    num_completions: int = 1,
    server_port: int = 30000,  # only for local server hosted on SGLang
    server_address: str = "localhost",
    server_type: str = "sglang",
    model_name: str = "default",  # specify model type

    # for reasoning models
    is_reasoning_model: bool = False, # indiactor of using reasoning models
    budget_tokens: int = 0, # for claude thinking
    reasoning_effort: str = None, # only for o1 and o3 / more reasoning models in the future
    max_completion_tokens: int = 4096,
):
    """
    Query various sort of LLM inference API providers
    Supports:
    - OpenAI
    - Deepseek
    - Together
    - Sambanova
    - Anthropic
    - Gemini / Google AI Studio
    - Fireworks (OpenAI compatbility)
    - Kimi / Moonshot AI (OpenAI compatibility)
    - SGLang (Local Server)
    """
    # Select model and client based on arguments
    match server_type:
        case "sglang":
            url = f"http://{server_address}:{server_port}"
            client = OpenAI(
                api_key=SGLANG_KEY, base_url=f"{url}/v1", timeout=None, max_retries=0
            )
            model = "default"
        case "deepseek":
            client = OpenAI(
                api_key=DEEPSEEK_KEY,
                base_url="https://api.deepseek.com",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name
            assert model in ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"], "Only support deepseek-chat or deepseek-coder for now"
            if not is_safe_to_send_to_deepseek(prompt):
                raise RuntimeError("Prompt is too long for DeepSeek")
        case "fireworks":
            client = OpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
                timeout=10000000,
                max_retries=3,
            )
            model = model_name

        case "anthropic":
            if not ANTHROPIC_KEY:
                raise ValueError("ANTHROPIC_API_KEY must be set when server_type='anthropic'")
            client = anthropic.Anthropic(
                api_key=ANTHROPIC_KEY,
            )
            model = model_name
        case "google":
            genai.configure(api_key=GEMINI_KEY)
            model = model_name
        case "together":
            client = Together(api_key=TOGETHER_KEY)
            model = model_name
        case "sambanova":
            client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
            model = model_name
        case "openai":
            client = OpenAI(api_key=OPENAI_KEY)
            model = model_name
        case "kimi":
            if not MOONSHOT_API_KEY:
                raise ValueError("MOONSHOT_API_KEY must be set when server_type='kimi'")
            client = OpenAI(
                api_key=MOONSHOT_API_KEY,
                base_url="https://api.moonshot.ai/v1",
                timeout=KIMI_TIMEOUT_SECONDS,
                max_retries=3,
            )
            model = model_name
            if model != "kimi-k3":
                raise ValueError("Kimi provider currently supports model_name='kimi-k3'")
        case "percepta":
            client = OpenAI(api_key=PERCEPTA_API_KEY, base_url="http://3.15.208.99:4000/v1")
            model = model_name
        case _:
            raise NotImplementedError

    if server_type != "google":
        assert client is not None, "Client is not set, cannot proceed to generations"

    print(
        f"Querying {server_type} {model} with temp {temperature} max tokens {max_tokens if not is_reasoning_model else max_completion_tokens}"
    )
    # Logic to query the LLM
    if server_type == "anthropic":
        assert type(prompt) == str
        anthropic_max_tokens = (
            max_completion_tokens if is_reasoning_model else max_tokens
        )
        messages = [{"role": "user", "content": prompt}]

        if is_reasoning_model and (
            _uses_adaptive_thinking(model) or reasoning_effort is not None
        ):
            # Sonnet 5 / recent Opus: adaptive thinking + effort.
            # max_tokens caps thinking + final text combined.
            request_kwargs = {
                "model": model,
                "system": system_prompt,
                "messages": messages,
                "max_tokens": anthropic_max_tokens,
                "thinking": {"type": "adaptive"},
                "output_config": {
                    "effort": reasoning_effort or "high",
                },
            }
            response = _anthropic_create_message(client, **request_kwargs)
        elif is_reasoning_model:
            # Legacy Claude reasoning path with fixed thinking budget.
            response = client.beta.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                max_tokens=anthropic_max_tokens,
                thinking={"type": "enabled", "budget_tokens": budget_tokens},
                betas=["output-128k-2025-02-19"],
            )
        elif _uses_adaptive_thinking(model):
            # Sonnet 5 rejects non-default temperature/top_p/top_k.
            response = _anthropic_create_message(
                client,
                model=model,
                system=system_prompt,
                messages=messages,
                max_tokens=anthropic_max_tokens,
            )
        else:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=anthropic_max_tokens,
            )
        outputs = _anthropic_text_outputs(response)
        if not outputs:
            print(
                "[WARN] Anthropic returned no text blocks "
                "(response may have been truncated by max_tokens while thinking)."
            )
            outputs = [""]

    elif server_type == "google":
        # assert model_name == "gemini-1.5-flash-002", "Only test this for now"
        try:
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_tokens,
                "response_mime_type": "text/plain",
            }
            
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config=generation_config,
            )

            response = model.generate_content(prompt)
            outputs = [response.text]
        except:
            return "Google model failed, try again"

    elif server_type == "deepseek":
        
        if model in ["deepseek-chat", "deepseek-coder"]:
            # regular deepseek model 
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )

        else: # deepseek reasoner
            assert is_reasoning_model, "Only support deepseek-reasoner for now"
            assert model == "deepseek-reasoner", "Only support deepseek-reasoner for now"
            response = client.chat.completions.create(
                    model=model,
                    messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                n=num_completions,
                max_tokens=max_tokens,
                # do not use temperature or top_p
            )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "openai":
        if is_reasoning_model:
            print(f"Using OpenAI reasoning model: {model} with reasoning effort {reasoning_effort}")
            request_kwargs = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": max_completion_tokens,
            }
            if reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**request_kwargs)
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "kimi":
        messages = (
            prompt
            if isinstance(prompt, list)
            else [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        request_kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": max_completion_tokens,
        }
        if reasoning_effort is not None:
            # Send the field at the top level while retaining compatibility
            # with OpenAI SDK releases that do not expose it explicitly.
            request_kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        response = client.chat.completions.create(**request_kwargs)
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "together":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            top_p=top_p,
            top_k=top_k,
            # repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            # truncate=32256,
            stream=False,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "fireworks":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            # top_p=top_p,
            # top_k=top_k,
            # repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            # truncate=32256,
            stream=False,
        )
        outputs = [choice.message.content for choice in response.choices]
    elif server_type == "sambanova":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            top_p=top_p,
        )
        outputs = [choice.message.content for choice in response.choices]
    # for all other kinds of servers, use standard API
    elif server_type == "percepta":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs = [choice.message.content for choice in response.choices]
    else:
        if type(prompt) == str:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.text for choice in response.choices]
        else:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                n=num_completions,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            outputs = [choice.message.content for choice in response.choices]

    logged_model = model if isinstance(model, str) else model_name
    log_usage(response, server_type, logged_model)

    # output processing
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


class LLMConfig:
    def __init__(self, server_type, model_name, **kwargs):
        self.server_type = server_type
        self.model_name = model_name
        # Save all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        # Convert all attributes to a dictionary
        result = {
            "server_type": self.server_type,
            "model_name": self.model_name
        }
        # Add any other attributes that might be added later
        for attr, value in self.__dict__.items():
            if attr not in result:
                if isinstance(value, (str, int, float, bool, type(None))):
                    result[attr] = value
                elif isinstance(value, list) or isinstance(value, dict):
                    result[attr] = value
                else:
                    pass
        return result

def create_llm_server_from_config(llm_config: LLMConfig = None, 
                                    greedy_sample: bool = False,   
                                    verbose: bool = False,
                                    time_generation: bool = False,
                                    # **kwargs,
                                ) -> callable:
    
    """
    Return a callable function that queries LLM with given settings
    """
    if llm_config is None:
        raise ValueError("llm_config is required")
    
    def _query_llm(prompt: str | list[dict]):

        # if kwargs:
        #     llm_config.update(kwargs)
        if greedy_sample:
            llm_config.temperature = 0.0
            llm_config.top_p = 1.0
            llm_config.top_k = 1
        if verbose:
            print(f"Querying server {llm_config.server_type} with args: {llm_config.to_dict()}")
        
        if time_generation:
            start_time = time.time()
            response = query_server(
                prompt, **llm_config.to_dict()
            )
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return query_server(
                prompt, **llm_config.to_dict()
            )
    
    return _query_llm

def make_dspy_lm(llm_config: LLMConfig) -> dspy.LM:
    temperature = float(llm_config.temperature)
    max_tokens = int(llm_config.max_completion_tokens) if bool(llm_config.is_reasoning_model) else int(llm_config.max_tokens)
    if llm_config.server_type == "openai" and bool(llm_config.is_reasoning_model):
        temperature = max(1.0, temperature)
        max_tokens = max(20000, max_tokens)
    return dspy.LM(f"{llm_config.server_type}/{llm_config.model_name}", temperature=temperature, max_tokens=max_tokens)
