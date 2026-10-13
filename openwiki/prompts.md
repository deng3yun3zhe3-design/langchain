---
type: "Concept"
title: "Prompts and Template Systems"
description: "LangChain's prompt system provides templates for string and chat models with variable substitution, composition, few-shot examples, and dynamic example selection. PromptValue abstractions bridge templates and models."
tags: [prompt, template, few-shot, example-selection, variable-substitution, structured-output, prompt-value, runnable]
sources:
  - id: openwiki-source-1f4e0a5b877db4f050f2a34c
    resource: repo://libs/core/langchain_core/example_selectors/base.py
  - id: openwiki-source-d533a177a8d9a5dd46f561d9
    resource: repo://libs/core/langchain_core/example_selectors/length_based.py
  - id: openwiki-source-5e027af8cc764d2750129cf1
    resource: repo://libs/core/langchain_core/example_selectors/semantic_similarity.py
  - id: openwiki-source-269af2bb49a21bd6c548b3f0
    resource: repo://libs/core/langchain_core/prompt_values.py
  - id: openwiki-source-03d7415879ed05a392edd62d
    resource: repo://libs/core/langchain_core/prompts/base.py
  - id: openwiki-source-15fdd645c1ee76ae559799c1
    resource: repo://libs/core/langchain_core/prompts/chat.py
  - id: openwiki-source-bc32774051e0e8a931a6fecd
    resource: repo://libs/core/langchain_core/prompts/few_shot.py
  - id: openwiki-source-c90241f4e6facb853f05d8e4
    resource: repo://libs/core/langchain_core/prompts/image.py
  - id: openwiki-source-5549894302ea4dfd5b8f4278
    resource: repo://libs/core/langchain_core/prompts/prompt.py
  - id: openwiki-source-cf81d0ba0a387a7cd9b5dfb8
    resource: repo://libs/core/langchain_core/prompts/string.py
  - id: openwiki-source-204b5e61a019044332bd2dd4
    resource: repo://libs/core/langchain_core/prompts/structured.py
generated: { by: "openwiki/0.5.0", at: "2026-10-13T08:31:57.014Z" }
verified:
  - by: openwiki/0.5.0
    at: 2026-10-13T08:31:57.014Z
---

## Overview

LangChain's **prompt templating system** provides a flexible, composable way to construct messages for language models. Prompts accept input variables, format them into message sequences, and optionally parse structured output. The system distinguishes between **string templates** (for raw text) and **chat templates** (sequences of typed messages). **Few-shot prompt templates** add the capability to select and inject examples dynamically, teaching models by demonstration.

## Prompt Values: The Core Abstraction

All prompts, regardless of type, produce a **PromptValue** when invoked. A `PromptValue` is the rendered output of a template, representing the formatted prompt ready for a language model. It acts as a bridge between templates and models, converting abstract templates into concrete inputs.

**PromptValue Types:**

- **StringPromptValue**: Contains a single text string, typically from `PromptTemplate`. The `to_messages()` method wraps the text in a `HumanMessage` for compatibility with chat models.
- **ChatPromptValue**: Contains a list of `BaseMessage` objects (system, human, AI, tool, etc.), returned by `ChatPromptTemplate`. The `to_string()` method concatenates messages into a single buffer string for non-chat APIs.
- **ImagePromptValue**: Contains an `ImageURL` dictionary with `url` and optional `detail` fields, returned by `ImagePromptTemplate` for multimodal models.

Both abstract methods are implemented on all types:
- `to_string() -> str`: Converts the prompt to a plain text string (e.g., via message concatenation for chat prompts).
- `to_messages() -> list[BaseMessage]`: Converts the prompt to a message list (e.g., wrapping text in a HumanMessage).

This design allows downstream components (LLMs, chat models, agents) to work with any prompt type uniformly:

```python
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# String template produces StringPromptValue
string_prompt = PromptTemplate.from_template("Hello {name}")
string_value = string_prompt.invoke({"name": "Alice"})
print(string_value.to_string())    # "Hello Alice"
print(string_value.to_messages())  # [HumanMessage(content="Hello Alice")]

# Chat template produces ChatPromptValue
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "Hello {name}")
])
chat_value = chat_prompt.invoke({"name": "Bob"})
print(chat_value.to_string())      # "System: You are helpful.\nHuman: Hello Bob"
print(chat_value.to_messages())    # [SystemMessage(...), HumanMessage(...)]
```

## Fundamental Concepts

### Prompt Types

LangChain provides three main categories of prompts:

#### PromptTemplate (String Prompts)

A `PromptTemplate` is a string template with variable placeholders that produces a `StringPromptValue` when formatted. It's used for simple text-based prompts or as building blocks within chat templates. The template is formatted using one of three engines:

- **f-string** (default): Python f-string syntax. Fast, supports arbitrary expressions in `{...}` brackets with proper escaping via `{{` and `}}`.
- **mustache**: Mustache syntax using `{{variable}}`. Safer for user-controlled templates.
- **jinja2**: Full Jinja2 templating. Supports conditionals, loops, and filters, but poses security risks if templates come from untrusted sources; LangChain uses `SandboxedEnvironment` by default for defense-in-depth.

**Key properties:**
- `template`: The template string.
- `input_variables`: List of variable names that must be provided during formatting.
- `partial_variables`: Pre-filled variables; reduce required inputs when formatting.
- `template_format`: Which engine to use (`f-string`, `mustache`, or `jinja2`).

```python
from langchain_core.prompts import PromptTemplate

# Simple f-string prompt
prompt = PromptTemplate.from_template("Tell me about {topic}")
output = prompt.format(topic="machine learning")

# Jinja2 with conditionals
prompt = PromptTemplate(
    template="{% if detailed %}Detailed:{% endif %} {content}",
    template_format="jinja2",
    input_variables=["content"],
    partial_variables={"detailed": True}
)

# Partial variables reduce required inputs
prompt = PromptTemplate(
    template="User: {name}, Topic: {topic}",
    input_variables=["topic"],
    partial_variables={"name": "Alice"}
)
result = prompt.format(topic="AI")  # name is already set
```

#### ChatPromptTemplate

A `ChatPromptTemplate` sequences message prompt templates into a conversation structure. Each message has a role (system, human, ai, tool, etc.) and content. This aligns with the message-based API of chat models like GPT-4 and Claude.

**Constructor patterns:**

```python
from langchain_core.prompts import ChatPromptTemplate

# Tuple shorthand
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, {name}"),
    ("ai", "Hi {name}! How can I help?"),
    ("human", "{user_input}"),
])

# Or direct list construction
template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "Hello, {name}"),
])
```

Supported message types in the shorthand syntax:
- `"system"` → `SystemMessagePromptTemplate`
- `"human"` → `HumanMessagePromptTemplate`
- `"ai"` → `AIMessagePromptTemplate`
- `"user"` → Alias for `"human"`
- `"assistant"` → Alias for `"ai"`
- `"tool"` / `"function"` → `ToolMessagePromptTemplate` / `FunctionMessagePromptTemplate`
- `"placeholder"` → `MessagesPlaceholder` for dynamic message lists

**Key methods:**
- `format_messages(**kwargs)`: Returns a list of `BaseMessage` objects.
- `invoke(dict)`: Runnable interface, returns `ChatPromptValue` containing formatted messages.
- `format(**kwargs)`: Converts message list to a single string (useful for debugging or non-chat APIs).

#### MessagesPlaceholder

A `MessagesPlaceholder` injects a pre-formatted list of messages at a specific point in the prompt. This is essential for maintaining conversation history.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),  # optional=True allows empty list
    ("human", "{question}"),
])

# Pass conversation history
result = template.invoke({
    "chat_history": [
        ("human", "What's 2+2?"),
        ("ai", "4"),
    ],
    "question": "And 3+3?",
})
# Messages: [system, human, ai, human]
```

The `optional=True` flag allows the placeholder to be omitted from inputs; if not provided, an empty list is substituted. The `n_messages` parameter limits how many recent messages are included (useful for token budgets).

### Template Variable Substitution

Prompts automatically detect variable names from template syntax and require them at format time.

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Q: {question}\nA: {answer}")
print(prompt.input_variables)  # ['answer', 'question']

# Variables are validated at runtime
try:
    prompt.format(question="What?")  # Missing 'answer'
except KeyError as e:
    print(f"Error: {e}")
```

**Partial application** pre-fills some variables, reducing the required input set:

```python
prompt = PromptTemplate.from_template("User: {name}, Question: {question}")
partial_prompt = prompt.partial(name="Bob")  # Bind 'name'
output = partial_prompt.format(question="How are you?")
# Only 'question' is required now
```

When a prompt has exactly **one** input variable, the template can accept a non-dict argument directly:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a bot."),
    ("human", "{input}"),
])
result = template.invoke("Hello!")  # Auto-wraps as {"input": "Hello!"}
```

#### ImagePromptTemplate (Multimodal Prompts)

An `ImagePromptTemplate` produces an `ImagePromptValue` containing image URLs or base64-encoded image data in OpenAI format. This is used for multimodal models that accept image inputs.

**Key features:**
- Uses a dictionary-based template with f-string, mustache, or jinja2 formatting
- Produces `ImagePromptValue` with `ImageURL` containing `url` and optional `detail` fields (auto, low, high)
- Integrates seamlessly with `ChatPromptTemplate` for models supporting image content blocks

```python
from langchain_core.prompts import ImagePromptTemplate, ChatPromptTemplate

# Image template with variable substitution
image_prompt = ImagePromptTemplate(
    input_variables=["image_id"],
    template={"url": "https://api.example.com/image/{image_id}.png", "detail": "high"},
    template_format="f-string",
)

image_value = image_prompt.invoke({"image_id": "cat123"})
# ImagePromptValue(image_url={"url": "https://api.example.com/image/cat123.png", "detail": "high"})

# Use in a chat template with text and image
template = ChatPromptTemplate.from_messages([
    ("system", "You are a vision expert."),
    ("human", [
        {"type": "text", "text": "Describe this image: {description}"},
        {"type": "image_url", "image_url": {"url": "{image_url}", "detail": "high"}},
    ]),
])

result = template.invoke({
    "description": "What's in this picture?",
    "image_url": "https://example.com/image.png",
})
```

## Few-Shot Prompt Templates

Few-shot learning teaches models by providing input-output examples before the user's actual query. LangChain provides two patterns: one for string prompts and one for chat-based prompts.

### FewShotPromptTemplate

`FewShotPromptTemplate` formats examples into a single string prompt.

**Structure:**
```
[prefix]

[formatted example 1]

[formatted example 2]

...

[suffix]
```

**Components:**
- `prefix`: Text before examples (optional).
- `example_prompt`: A `PromptTemplate` specifying how each example is formatted.
- `examples` or `example_selector`: Source of examples (either a fixed list or dynamic selector).
- `suffix`: Text after examples. Usually contains the actual task and placeholders for the new input.
- `example_separator`: String joining prefix, examples, and suffix (default: `"\n\n"`).

```python
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

output = prompt.format(input="big")
# Output:
# Input: happy
# Output: sad
#
# Input: tall
# Output: short
#
# Input: big
# Output:
```

### FewShotChatMessagePromptTemplate

`FewShotChatMessagePromptTemplate` embeds examples as message pairs within a chat sequence.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "What is {input}?"),
    ("ai", "{output}"),
])

few_shot = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful math tutor."),
    few_shot,
    ("human", "What is {input}?"),
])

result = template.invoke({"input": "4+4"})
# Messages: [system, human(2+2?), ai(4), human(2+3?), ai(5), human(4+4?)]
```

## Example Selectors

Instead of using a fixed list of examples, an **example selector** dynamically picks relevant examples based on the input. This optimizes prompt length and relevance.

### BaseExampleSelector Interface

All selectors implement:

```python
class BaseExampleSelector:
    def add_example(self, example: dict[str, str]) -> Any:
        """Add a new example to the store."""
        
    def select_examples(self, input_variables: dict[str, str]) -> list[dict[str, Any]]:
        """Select which examples to use based on inputs."""
```

### SemanticSimilarityExampleSelector

Embeds examples and input into a vector space, retrieving the `k` most similar examples. Requires a `VectorStore` and embeddings model.

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
]

# Create vector store from example texts
to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,  # Always return 2 examples
)

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# When formatting, the selector retrieves the 2 most similar examples to "bright"
output = prompt.format(input="bright")
```

**Key parameters:**
- `vectorstore`: VectorStore containing embedded examples.
- `k`: Number of examples to return (default: 4).
- `input_keys`: Optional filter to use only specific keys for similarity search (e.g., only the "input" field, not "output").
- `example_keys`: Optional filter to include only certain keys in returned examples.
- `vectorstore_kwargs`: Extra arguments passed to the vectorstore's `similarity_search` method.

### LengthBasedExampleSelector

Selects examples greedily up to a maximum token/word count, preventing prompt length overflow. Useful when token budgets are tight.

```python
from langchain_core.example_selectors import LengthBasedExampleSelector
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
]

example_prompt = PromptTemplate(
    template="Input: {input}\nOutput: {output}",
    input_variables=["input", "output"],
)

selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50,  # Limit prompt to ~50 words
    get_text_length=lambda x: len(x.split()),  # Custom length function
)

prompt = FewShotPromptTemplate(
    example_selector=selector,
    example_prompt=example_prompt,
    suffix="Input: {input}\nOutput:",
    input_variables=["input"],
)

# Selector returns only as many examples as fit within max_length
output = prompt.format(input="fast")
```

**Key parameters:**
- `examples`: List of all available examples.
- `max_length`: Maximum prompt length (tokens or words, determined by `get_text_length`).
- `get_text_length`: Function to measure prompt length; defaults to word count via regex.

**Behavior:** Examples are iterated in order; the selector stops adding when the next example would exceed `max_length`. This is greedy, not optimal, but fast and predictable.

## Structured Output Prompts

The `StructuredPrompt` (beta) combines a `ChatPromptTemplate` with a Pydantic schema, enabling the model to produce JSON output matching a specific schema.

```python
from pydantic import BaseModel
from langchain_core.prompts import StructuredPrompt

class QuestionAnswer(BaseModel):
    question: str
    answer: str

template = StructuredPrompt.from_messages_and_schema(
    messages=[
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
    ],
    schema=QuestionAnswer,
)

# When invoked with a model supporting structured output,
# the model is instructed to return JSON matching QuestionAnswer
result = template.invoke({"input": "What is LangChain?"})
```

This is useful for tasks requiring consistent, parseable output (e.g., fact extraction, data classification).

## Runnable Interface Specification

All prompts inherit from `RunnableSerializable`, implementing the LangChain runnable protocol with these core methods:

- `invoke(input: dict) -> PromptValue`: Synchronous formatting; returns `StringPromptValue`, `ChatPromptValue`, or `ImagePromptValue`.
- `ainvoke(input: dict) -> PromptValue`: Asynchronous variant using `async def`.
- `batch(inputs: list[dict], config: RunnableConfig | None = None) -> list[PromptValue]`: Format multiple inputs in parallel or sequence.
- `stream(input: dict) -> Iterator[PromptValue]`: Stream the output (rarely used for prompts; streaming is more common downstream in model outputs).

**Metadata and Configuration:**

All prompts support metadata tracking for observability:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# Attach metadata for tracing and debugging
template_with_meta = template.with_config({
    "metadata": {"version": "1.0", "owner": "team-ai"},
    "tags": ["production", "english"],
})

result = template_with_meta.invoke({"input": "hello"})
# Metadata is propagated to LangSmith, callbacks, and tracing backends
```

## Prompt Composition

Prompts compose via two mechanisms: the `+` operator for template merging, and the `|` (pipe) operator for runnable chains.

### Template Merging with `+`

The `+` operator concatenates message sequences and merges input variables:

```python
from langchain_core.prompts import ChatPromptTemplate

system = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Your name is {bot_name}."),
])

conversation = ChatPromptTemplate.from_messages([
    ("human", "{user_input}"),
])

combined = system + conversation
# Equivalent to:
# ChatPromptTemplate([
#     ("system", "You are a helpful assistant. Your name is {bot_name}."),
#     ("human", "{user_input}"),
# ])

result = combined.invoke({"bot_name": "Alice", "user_input": "Hello!"})
```

**Rules for `+` operator:**
- When combining `ChatPromptTemplate` instances, messages are concatenated.
- Input variables from both templates are merged.
- Partial variables are merged; conflicting keys raise an error.
- Templates must have compatible formats (both f-string, both mustache, etc.).

### Chaining with `|` (Pipe)

Prompts integrate with LangChain's runnable interface, enabling pipes (`|`) to chain prompts with models and parsers:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{query}"),
])

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = template | model | parser

# Synchronous execution
result = chain.invoke({"query": "What is LangChain?"})

# Batch processing
results = chain.batch([{"query": "Q1"}, {"query": "Q2"}])

# Streaming
for chunk in chain.stream({"query": "Tell a story"}):
    print(chunk, end="", flush=True)

# Async execution
async_result = await chain.ainvoke({"query": "Explain recursion"})
```

The pipe operator works because prompts are `RunnableSerializable` instances that implement the runnable protocol.

## Prompt Loading from Files

**Note:** Prompt serialization and loading via the old `save()` / `load_prompt_from_config()` API is deprecated in favor of using `dumpd()` / `loads()` from `langchain_core.load`.

To load a prompt from a JSON or YAML file, use the modern LangChain serialization API:

```python
from langchain_core.load import loads
import json

with open("prompt.json") as f:
    prompt_dict = json.load(f)

prompt = loads(prompt_dict)
# Returns a deserialized PromptTemplate or ChatPromptTemplate
```

Prompts can be serialized to JSON using `dumpd()` from `langchain_core.load`, enabling version control and sharing:

```python
from langchain_core.load import dumpd
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

prompt_dict = dumpd(prompt)
# Contains nested structure compatible with loads()
```

## Integration with Agent Factory

Prompts are a core input to the **Agent Factory** (`create_agent`), providing the conversational context for agent reasoning and tool use.

```python
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

system_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful weather assistant."),
])

# Or use a simple string
agent = create_agent(
    model="openai:gpt-4o",
    tools=[weather_tool],
    system_prompt="You are a helpful weather assistant."
)
```

The agent factory internally compiles prompts with the model and tool bindings, managing message flow through the state machine. Middleware can intercept and modify prompts before model invocation via the `wrap_model_call` hook, enabling use cases like prompt optimization or safety filters.

## Security Considerations

### Template Injection

When constructing prompts from user input, use **partial variables** or **input variables** instead of string concatenation:

```python
# UNSAFE: Vulnerable to prompt injection
user_input = input("Enter text: ")
template = f"User said: {user_input}"  # Don't do this

# SAFE: Use variable substitution
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("User said: {user_input}")
output = prompt.format(user_input=user_input)
```

### Jinja2 Sandboxing

When using Jinja2 templates, LangChain applies `SandboxedEnvironment` by default, blocking access to dunder attributes (`__class__`, `__globals__`, etc.). However:

- **Do not accept Jinja2 templates from untrusted sources.** Sandboxing is best-effort, not foolproof.
- Regular method calls and attribute access are still allowed (e.g., `obj.method()`).
- If you must use Jinja2, prefer `f-string` or `mustache` for untrusted inputs.

```python
# Safe: f-string template from user, validated at construction time
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="Hello {name}",  # User-provided, but simple variable syntax
    input_variables=["name"],
)
```

## Lifecycle and State

Prompt templates are **immutable** in the functional sense: calling `format()` or `invoke()` does not mutate the template. Methods like `partial()` return new instances.

```python
original = PromptTemplate.from_template("Say {text}")
partial = original.partial(text="hello")  # Returns a NEW PromptTemplate

# original is unchanged
print(original.input_variables)  # ['text']
print(partial.input_variables)   # []
```

This immutability enables safe composition and caching in pipelines.

## Observability and Tracing

All prompts support LangChain's standard tracing and observability hooks:

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# Add metadata for tracing
template_with_metadata = template.with_config({
    "metadata": {"version": "1.0"},
    "tags": ["important"],
})

result = template_with_metadata.invoke({"input": "hello"})
# The invoke is traced with the given metadata
```

Metadata and tags are propagated to LangSmith and other observability backends, enabling debugging and performance analysis.
