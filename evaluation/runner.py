"""Agent-loop runner: sends question to LLM, executes code, collects answer."""
import json
import re
import time
import httpx
from openai import OpenAI
from anthropic import Anthropic

from .config import DEFAULT_MODEL, MAX_TURNS, get_client_config, MODEL_PROVIDERS
from . import executor as executor_module
from .auto_scorer import parse_answer_block
from .executor import execute_python, isolated_signal_workspace


def _extract_dsml_code(content):
    """Extract Python code from DSML-format tool calls in content."""
    if not content or 'DSML' not in content:
        return None
    m = re.search(
        r'DSML.*?parameter\s+name="code"[^>]*>\s*\n?(.*)',
        content, re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None


def _extract_markdown_code(content):
    """Extract Python code from markdown code blocks (for models that don't use tool_calls)."""
    if not content:
        return None
    blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', content, re.DOTALL)
    if blocks:
        return '\n\n'.join(b.strip() for b in blocks)
    return None


def _extract_kimi_tool_code(content):
    """Extract Python code from Kimi's raw tool token format."""
    if not content or '<|tool_call' not in content:
        return None
    m = re.search(
        r'functions\.execute_python:\d+\s*<\|tool_call_argument_begin\|>\s*(\{.*?\})\s*<\|tool_call_end\|>',
        content, re.DOTALL,
    )
    if not m:
        m = re.search(
            r'functions\.execute_python:\d+\s*<\|tool_call_argument_begin\|>\s*(\{.*)',
            content, re.DOTALL,
        )
    if m:
        try:
            args = json.loads(m.group(1))
            return args.get("code", "")
        except json.JSONDecodeError:
            code_m = re.search(r'"code"\s*:\s*"(.*)"', m.group(1), re.DOTALL)
            if code_m:
                return code_m.group(1)
    return None


def _is_anthropic_provider(model):
    if model in MODEL_PROVIDERS:
        provider = MODEL_PROVIDERS[model][0]
        return provider == "anthropic"
    return False


SYSTEM_PROMPT = """\
You are an expert in electromagnetic signal analysis. You will analyze raw I/Q signal data and answer questions.

## Tools
You can call the execute_python tool to run Python code locally.
Available libraries: numpy, scipy, matplotlib, sklearn, etc.
Signal files (.npy) are in the current working directory — load with np.load('filename.npy').

## Strategy
1. First call: load signal, compute FFT/PSD, print all signals' frequencies, bandwidths, powers.
2. Subsequent calls: targeted analysis for specific sub-questions.
3. Combine analysis steps to minimize tool calls (target: 3-6 calls total).
4. Once you have enough data, give your final answer immediately — do not over-analyze.

## Output Format
After completing your analysis, end your response with a structured answer block:

===ANSWERS===
Q1a: <value> <unit>
Q1b: <value> <unit>
Q1c: ...
Q2a: ...
===END===

Include answers for ALL sub-questions. If the user question provides a stricter JSON schema inside this block, follow that schema exactly instead of the generic label format above."""

LEGACY_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Include answers for ALL sub-questions. If the user question provides a "
    "stricter JSON schema inside this block, follow that schema exactly "
    "instead of the generic label format above.",
    "Include answers for ALL sub-questions. For qualitative questions, give "
    "a brief text answer.",
)

DEEPSEEK_FORCE_ANSWER_ADDENDUM = (
    "\n\nIMPORTANT: Do NOT write any code. Do NOT use tool calls. "
    "Do NOT use DSML tags. Output ONLY plain text in the "
    "===ANSWERS=== format above."
)


def _force_answer_prompts(question_text):
    if 'emrb-l4-repaired-v1' in question_text:
        return [
            ("Stop further code analysis and output the final answer now. "
             "Use exactly one Q1 through Q5 line inside ===ANSWERS=== and "
             "===END===. For any question that requests a JSON object, place "
             "one valid single-line JSON object directly after its Q label."),
            ("The previous response did not follow the required format. Return "
             "only ===ANSWERS===, five lines labeled Q1 through Q5, and "
             "===END===. Preserve every JSON field requested by the question."),
        ]
    if 'emrb-l2-autocorr-v1' in question_text:
        return [
            ("Stop further code analysis and output the final answers NOW.\n"
             "Format:\n===ANSWERS===\nQ1a: ...\nQ1b: ...\n...\n===END===\n"
             "Answer every sub-question of Q1, Q2, Q4, and Q5 on its own "
             "labeled line. For Q3, output exactly one line 'Q3:' followed by "
             "the single-line JSON object requested by the question, with "
             "every field filled in."),
            ("The previous response did not follow the required format. Do NOT "
             "write code. Return only ===ANSWERS===, one labeled line per "
             "sub-question of Q1/Q2/Q4/Q5, one single-line JSON object on the "
             "Q3 line containing every field requested by the question, and "
             "===END==="),
        ]
    schema_match = re.search(r'emrb-l5-verifiable-v\d+', question_text)
    if schema_match:
        schema_version = schema_match.group(0)
        return [
            ("Stop further code analysis and output the final answer now. "
             "Use the exact JSON schema printed at the end of the question, "
             "replace every null, and place the valid JSON between "
             "===ANSWERS=== and ===END===. Do not write code or omit fields."),
            ("The previous response did not follow the required format. "
             "Return only ===ANSWERS===, one valid JSON object matching the "
             f"question's {schema_version} schema, and ===END===."),
        ]
    return [
        ("Stop further code analysis. Based on all results so far, "
         "output your final answers NOW.\n"
         "Format:\n===ANSWERS===\nQ1a: ...\nQ1b: ...\n...\n===END===\n"
         "You MUST provide your best estimate for every sub-question."),
        ("You did NOT follow the required format. Do NOT write code. "
         "Do NOT think further. Just output answers IMMEDIATELY.\n"
         "Copy this template and fill in values — one line per sub-question, "
         "including EVERY lettered part that the question asked:\n\n"
         "===ANSWERS===\n"
         + '\n'.join(f"Q{n}{letter}: [value]"
                     for n in range(1, 6) for letter in 'abcd')
         + "\n===END==="),
    ]


_QUESTION_HEADER_RE = re.compile(r'(?m)^\s*(Q\d+)\s*[.:：]')
_SUBQUESTION_HEADER_RE = re.compile(r'\(([a-d])\)\s*', re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r'''^\s*(?:
        <[^>]+>
        | \[[^]]*value[^]]*]
        | \.\.\.
        | n/?a
        | unknown
        | undetermined
        | not\s+(?:calculated|computed|available|provided|determined)
        | (?:unable|failed)\s+to\s+(?:calculate|compute|determine)
        | cannot\s+(?:calculate|compute|determine)
        | 未(?:计算|求出|确定)
        | 无法(?:计算|求出|确定)
        | 不(?:知道|确定)
        | 未知
        | 无答案
    )\s*[.!。！]?\s*$''', re.IGNORECASE | re.VERBOSE)


def _required_plain_answer_labels(question_text):
    headers = list(_QUESTION_HEADER_RE.finditer(question_text))
    required = set()
    for index, header in enumerate(headers):
        qid = header.group(1)
        end = headers[index + 1].start() if index + 1 < len(headers) \
            else len(question_text)
        section = question_text[header.start():end]
        letters = {
            match.group(1).lower()
            for match in _SUBQUESTION_HEADER_RE.finditer(section)
        }
        required.update(
            {f'{qid}{letter}' for letter in letters} if letters else {qid}
        )
    return required


def _answer_block_complete(question_text, content):
    """Whether the latest answer block is closed and covers the prompt.

    L1/L3 require one non-placeholder value for every lettered sub-question.
    Schema-bearing tasks must also pass the same structural validator used by
    their deterministic scorer.
    """
    content = str(content or '')
    if '===ANSWERS===' not in content:
        return False
    latest = content.rsplit('===ANSWERS===', 1)[1]
    if '===END===' not in latest:
        return False
    if ('emrb-l2-autocorr-v1' in question_text
            or 'emrb-l4-repaired-v1' in question_text
            or re.search(r'emrb-l5-verifiable-v\d+', question_text)):
        from evaluation.pipeline_runner import _structured_answer_ok
        return _structured_answer_ok(question_text, content)
    required = _required_plain_answer_labels(question_text)
    if not required:
        return True
    answers = parse_answer_block(content)
    def present(label):
        candidates = [answers[label]] if label in answers else []
        if re.fullmatch(r'Q\d+', label):
            candidates.extend(
                value for key, value in answers.items()
                if key.startswith(label) and key != label
            )
        return any(
            bool(str(value).strip())
            and not _PLACEHOLDER_RE.fullmatch(str(value))
            for value in candidates
        )

    return all(present(label) for label in required)


def _select_final_assistant_text(messages, question_text):
    candidates = []
    for message in messages:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        content = (message.get('content', '')
                   or message.get('reasoning_content', '') or '')
        if isinstance(content, str) and content.strip():
            candidates.append(content)
    for content in reversed(candidates):
        if _answer_block_complete(question_text, content):
            return content
    for content in reversed(candidates):
        if '===ANSWERS===' in content:
            return content
    return candidates[-1] if candidates else ''


def _clean_final_answer_prompt(question_text, messages, format_prompt):
    saved_outputs = []
    for message in messages:
        content = message.get("content", "") if isinstance(message, dict) else ""
        if message.get("role") == "tool" and isinstance(content, str):
            saved_outputs.append(content)
        elif (
            message.get("role") == "user"
            and isinstance(content, str)
            and content.startswith("Code execution result:\n")
        ):
            saved_outputs.append(content.removeprefix("Code execution result:\n"))

    evidence = "\n\n".join(
        f"--- SAVED TOOL OUTPUT {index}/{len(saved_outputs)} ---\n{output}"
        for index, output in enumerate(saved_outputs, start=1)
    )
    return (
        "ORIGINAL QUESTION\n"
        f"{question_text}\n\n"
        "SAVED RESULTS FROM THE COMPLETED PYTHON ANALYSIS\n"
        f"{evidence}\n\n"
        f"{format_prompt}\n"
        "Use the saved measurements above. Do not call tools, write code, "
        "or omit any requested sub-question."
    )


def _flatten_tool_history(messages):
    flattened = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls") or []

        if role == "assistant" and tool_calls:
            parts = [content] if content else []
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "")
                try:
                    parsed = json.loads(arguments)
                    code = parsed.get("code", arguments)
                except (TypeError, json.JSONDecodeError):
                    code = arguments
                parts.append(
                    "I ran this Python analysis:\n```python\n"
                    f"{code}\n```"
                )
            flattened.append({
                "role": "assistant",
                "content": "\n\n".join(parts),
            })
        elif role == "tool":
            flattened.append({
                "role": "user",
                "content": (
                    "Python execution result from the previous analysis:\n"
                    f"{content}"
                ),
            })
        else:
            flattened.append({"role": role, "content": content})
    return flattened


def _wire_messages(messages, profile):
    if profile['flatten_tool_history']:
        return _flatten_tool_history(messages)
    return messages

TOOLS_OPENAI = [{
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Execute Python code locally. numpy/scipy/matplotlib available. "
                       "Signal .npy files are in the current directory, load with np.load().",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
    },
}]

TOOLS_ANTHROPIC = [{
    "name": "execute_python",
    "description": "Execute Python code locally. numpy/scipy/matplotlib available. "
                   "Signal .npy files are in the current directory, load with np.load().",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"],
    },
}]


MODEL_NAME_MAP = {
    "minimax-m3": "MiniMax-M3",
    "llama-3.1-70b": "meta/llama-3.1-70b-instruct",
    "llama-3.3-70b": "meta/llama-3.3-70b-instruct",
}

PROVIDERS_NEEDING_STREAM = {"deepseek"}

FREEFORM_PROTOCOL_VERSION = 'free-form-v4'
LEGACY_FREEFORM_PROTOCOL_VERSION = 'legacy-free-form-v1'

REASONING_MODELS = frozenset({
    'gpt-5',
    'gpt-5.4',
    'gpt-5.5',
    'deepseek-reasoner',
    'deepseek-v4-pro',
    'deepseek-v4-flash',
    'o1',
    'o3',
    'o4-mini',
    'gemini-3.1-pro-preview',
    'gemini-3.1-pro',
    'gemini-3.5-flash',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'grok-4',
    'grok-4.5',
    'mimo-v2.5-pro',
    'hy3',
})

LEGACY_GENERIC_FORCE_PROMPTS = [
    ("Stop further code analysis. Based on all results so far, "
     "output your final answers NOW.\n"
     "Format:\n===ANSWERS===\nQ1a: ...\nQ1b: ...\n...\n===END===\n"
     "You MUST provide your best estimate for every sub-question."),
    ("You did NOT follow the required format. Do NOT write code. "
     "Do NOT think further. Just output answers IMMEDIATELY.\n"
     "Copy this template and fill in values:\n\n"
     "===ANSWERS===\nQ1a: [value]\nQ1b: [value]\nQ2a: [value]\n"
     "Q2b: [value]\nQ2c: [value]\nQ2d: [value]\nQ3a: [value]\n"
     "Q3b: [value]\nQ4a: [value]\nQ4b: [value]\nQ5a: [value]\n"
     "Q5b: [value]\n===END==="),
]


def model_execution_profile(model):
    """Resolved, credential-free model route and generation controls."""
    provider = MODEL_PROVIDERS.get(model, ('deepseek',))[0]
    _, base_url = get_client_config(model)
    transport = 'anthropic-native' if provider == 'anthropic' \
        else 'openai-compatible'
    api_model = model if transport == 'anthropic-native' \
        else MODEL_NAME_MAP.get(model, model)
    reasoning = model in REASONING_MODELS
    deepseek_v4 = model in {'deepseek-v4-pro', 'deepseek-v4-flash'}
    nonthinking_finalization = deepseek_v4
    reasoning_knob_supported = provider != 'deepseek'
    extended_finalization = (
        model in {'gemini-3.5-flash'}
        or deepseek_v4
    )
    return {
        'requested_model': model,
        'provider': provider,
        'base_url': base_url,
        'transport': transport,
        'api_model': api_model,
        'reasoning_model': reasoning,
        'api_timeout_s': None if transport == 'anthropic-native'
        else (900 if reasoning else 300),
        'generation_max_tokens': 16384 if transport == 'anthropic-native'
        else (40000 if reasoning else 4096),
        'force_answer_max_tokens': (
            16384 if extended_finalization else 4096
        ),
        'force_answer_attempts': 1 if transport == 'anthropic-native' else 2,
        'clean_final_answer_fallback': extended_finalization,
        'deepseek_v4': deepseek_v4,
        'flatten_tool_history': False,
        'temperature': 0.0,
        'reasoning_effort': None,
        'force_answer_reasoning_effort': (
            'none' if nonthinking_finalization and reasoning_knob_supported
            else None
        ),
        'force_answer_thinking': (
            False if nonthinking_finalization and reasoning_knob_supported
            else None
        ),
        'stream': (transport == 'openai-compatible'
                   and provider in PROVIDERS_NEEDING_STREAM),
        'clean_headers': False,
        'bypass_environment_proxy': False,
        'required_tool_cache_bust': False,
        'first_tool_choice': (
            'anthropic-default'
            if transport == 'anthropic-native'
            else ('auto' if deepseek_v4
                  else 'required')
        ),
        'later_tool_choice': (
            'anthropic-default'
            if transport == 'anthropic-native' else 'auto'
        ),
        'force_answer_tool_choice': (
            'no-tools'
            if transport == 'anthropic-native'
            or nonthinking_finalization
            else 'none'
        ),
    }


def _apply_force_answer_controls(request, profile):
    """Apply provider-compatible controls for answer-only generation."""
    if profile['force_answer_thinking'] is None:
        return
    request['reasoning_effort'] = profile['force_answer_reasoning_effort']
    request['extra_body'] = {
        'chat_template_kwargs': {
            'thinking': profile['force_answer_thinking'],
            'reasoning_effort': profile['force_answer_reasoning_effort'],
        },
    }


def _collect_stream(stream):
    """Collect streaming chunks into a pseudo ChatCompletion-like object."""
    content_parts = []
    tool_calls_map = {}  # index -> {id, name, arguments}
    finish_reason = None
    reasoning_parts = []

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
            reasoning_parts.append(delta.reasoning_content)
        if delta.content:
            content_parts.append(delta.content)
        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {
                        'id': tc_delta.id or '',
                        'name': '',
                        'arguments': '',
                    }
                entry = tool_calls_map[idx]
                if tc_delta.id:
                    entry['id'] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        entry['name'] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        entry['arguments'] += tc_delta.function.arguments

    class _Func:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _TC:
        def __init__(self, id, function):
            self.id = id
            self.function = function

    class _Msg:
        def __init__(self, content, tool_calls, reasoning_content):
            self.content = content
            self.tool_calls = tool_calls
            self.reasoning_content = reasoning_content

    class _Choice:
        def __init__(self, message, finish_reason):
            self.message = message
            self.finish_reason = finish_reason

    class _Resp:
        def __init__(self, choices):
            self.choices = choices

    tcs = None
    if tool_calls_map:
        tcs = []
        for idx in sorted(tool_calls_map):
            e = tool_calls_map[idx]
            tcs.append(_TC(e['id'], _Func(e['name'], e['arguments'])))

    content = ''.join(content_parts) or None
    reasoning = ''.join(reasoning_parts) or None
    msg = _Msg(content, tcs or None, reasoning)
    choice = _Choice(msg, finish_reason)
    return _Resp([choice])


def _cache_bust_wire_messages(messages, trailing_spaces):
    """Change a provider cache key without changing prompt semantics."""
    if trailing_spaces <= 0:
        return messages
    updated = []
    changed = False
    for message in messages:
        copied = dict(message)
        if (
            not changed
            and copied.get("role") == "system"
            and isinstance(copied.get("content"), str)
        ):
            copied["content"] += " " * trailing_spaces
            changed = True
        updated.append(copied)
    return updated


def _run_openai(question_text, sample_dir, model, max_turns, verbose):
    """Run agent loop using OpenAI-compatible API."""
    api_key, base_url = get_client_config(model)
    profile = model_execution_profile(model)
    api_model = profile['api_model']
    api_timeout = profile['api_timeout_s']
    client_kwargs = dict(api_key=api_key, base_url=base_url, timeout=api_timeout)
    if profile['clean_headers']:
        client_kwargs["default_headers"] = {"User-Agent": "python-httpx/0.27"}
    if profile['bypass_environment_proxy']:
        client_kwargs["http_client"] = httpx.Client(
            trust_env=False, timeout=api_timeout
        )
    client = OpenAI(**client_kwargs)
    gen_max_tokens = profile['generation_max_tokens']

    reasoning_effort = profile['reasoning_effort']

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]

    code_calls = 0
    retries = 0
    empty_retries = 0
    required_tool_retries = 0
    cache_bust_spaces = 1 if profile['required_tool_cache_bust'] else 0
    t0 = time.time()
    turn = 0

    def request_messages():
        wire_messages = _wire_messages(messages, profile)
        return _cache_bust_wire_messages(wire_messages, cache_bust_spaces)

    while turn < max_turns:
        turn += 1
        if verbose:
            print(f"  [turn {turn}/{max_turns}] calling {model}...", end=" ", flush=True)

        try:
            tc = (profile['first_tool_choice'] if code_calls == 0
                  else profile['later_tool_choice'])
            create_kwargs = dict(
                model=api_model,
                messages=request_messages(),
                tools=TOOLS_OPENAI,
                tool_choice=tc,
                max_tokens=gen_max_tokens,
                temperature=profile['temperature'],
            )
            if reasoning_effort:
                create_kwargs["reasoning_effort"] = reasoning_effort
            use_stream = profile['stream']
            if use_stream:
                create_kwargs["stream"] = True
                stream = client.chat.completions.create(**create_kwargs)
                resp = _collect_stream(stream)
            else:
                resp = client.chat.completions.create(**create_kwargs)
            if not hasattr(resp, 'choices'):
                raise ValueError(f"API returned non-standard response: {str(resp)[:200]}")
        except Exception as e:
            retries += 1
            wait = min(10 * (2 ** min(retries - 1, 4)), 120)
            if verbose:
                print(f"API error: {e}, retry #{retries} in {wait}s...")
            if retries > 10:
                if verbose:
                    print("Max retries exceeded, giving up on this problem.")
                break
            time.sleep(wait)
            turn -= 1
            continue

        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, 'reasoning_content', None)

        # This provider caches incomplete responses that retain the assistant
        # text but lose required tool_calls. Never turn such a cache hit into
        # a guessed answer. A different amount of trailing system whitespace
        # changes only the transport cache key and preserves prompt semantics.
        if (
            profile['required_tool_cache_bust']
            and code_calls == 0
            and not msg.tool_calls
        ):
            required_tool_retries += 1
            if required_tool_retries > 3:
                raise RuntimeError(
                    f"{model} omitted the required tool call after 4 "
                    "cache-busted attempts"
                )
            cache_bust_spaces += 1
            if verbose:
                print(
                    "required tool call missing; changing cache key and "
                    f"retrying ({required_tool_retries}/3)..."
                )
            turn -= 1
            continue

        # Some OpenAI-compatible routes occasionally return a successful HTTP
        # response with no text, reasoning, or tool call. Treat that as a
        # transport failure; accepting it would skip data analysis and force a
        # fabricated final answer from an empty transcript.
        if not msg.tool_calls and not (msg.content or reasoning):
            empty_retries += 1
            if empty_retries > 3:
                raise RuntimeError(
                    f"{model} returned 4 consecutive empty responses"
                )
            wait = min(5 * (2 ** (empty_retries - 1)), 30)
            if verbose:
                print(
                    f"empty response, retry #{empty_retries} in {wait}s..."
                )
            time.sleep(wait)
            turn -= 1
            continue

        retries = 0
        empty_retries = 0

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "execute_python":
                    try:
                        args = json.loads(tc.function.arguments)
                        code = args["code"]
                    except (json.JSONDecodeError, KeyError):
                        code = tc.function.arguments
                    if verbose:
                        first_line = code.strip().split('\n')[0][:60]
                        print(f"exec: {first_line}...")
                    result = execute_python(code, sample_dir)
                    code_calls += 1
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"Unknown tool: {tc.function.name}",
                    })
        else:
            dsml_code = _extract_dsml_code(msg.content)
            kimi_code = _extract_kimi_tool_code(msg.content) if not dsml_code else None
            md_code = _extract_markdown_code(msg.content) if not (dsml_code or kimi_code) else None
            inline_code = dsml_code or kimi_code or md_code
            if inline_code:
                tag = "dsml" if dsml_code else ("kimi" if kimi_code else "md")
                if verbose:
                    first_line = inline_code.strip().split('\n')[0][:60]
                    print(f"exec({tag}): {first_line}...")
                result = execute_python(inline_code, sample_dir)
                code_calls += 1
                messages.append({
                    "role": "user",
                    "content": f"Code execution result:\n{result}",
                })
                continue
            if verbose:
                print("done (no tool call)")
            break

        if choice.finish_reason == "stop" and not msg.tool_calls:
            if verbose:
                print("done (stop)")
            break

    # Force final answer if needed
    last_content = ""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content", "") or m.get("reasoning_content", "") or ""
            if c:
                last_content = c
                break

    if not _answer_block_complete(question_text, last_content):
        fa_prompts = _force_answer_prompts(question_text)
        if profile['deepseek_v4']:
            fa_prompts[0] += DEEPSEEK_FORCE_ANSWER_ADDENDUM
        for fa_attempt, fa_prompt in enumerate(fa_prompts):
            if verbose:
                print(f"  [final] Requesting structured answer (attempt {fa_attempt+1})...")
            messages.append({"role": "user", "content": fa_prompt})
            try:
                fa_kwargs = dict(
                    model=api_model,
                    messages=request_messages(),
                    max_tokens=profile['force_answer_max_tokens'],
                    temperature=profile['temperature'],
                )
                _apply_force_answer_controls(fa_kwargs, profile)
                if profile['force_answer_tool_choice'] == 'none':
                    fa_kwargs["tools"] = TOOLS_OPENAI
                    fa_kwargs["tool_choice"] = "none"
                if profile['stream']:
                    fa_kwargs["stream"] = True
                    resp = _collect_stream(client.chat.completions.create(**fa_kwargs))
                else:
                    resp = client.chat.completions.create(**fa_kwargs)
                if not hasattr(resp, 'choices'):
                    raise ValueError("Non-standard response in force-answer")
                final_msg = resp.choices[0].message
                final_content = final_msg.content or getattr(final_msg, 'reasoning_content', '') or ""
                if '<|tool_call' in final_content:
                    final_content = final_content.split('<|tool_call')[0].strip()
                if 'DSML' in final_content:
                    final_content = re.sub(r'<[｜\|]+DSML[｜\|]+.*', '', final_content, flags=re.DOTALL).strip()
                messages.append({"role": "assistant",
                                 "content": final_content})
                if _answer_block_complete(question_text, final_content):
                    break
            except Exception as e:
                if verbose:
                    print(f"  [final] Force-answer attempt {fa_attempt+1} failed: {e}")
                continue

    last_content = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = (
                message.get("content", "")
                or message.get("reasoning_content", "")
                or ""
            )
            if content:
                last_content = content
                break

    if (
        profile['clean_final_answer_fallback']
        and not _answer_block_complete(question_text, last_content)
    ):
        for fallback_attempt, format_prompt in enumerate(
            _force_answer_prompts(question_text), start=1
        ):
            clean_prompt = _clean_final_answer_prompt(
                question_text, messages, format_prompt
            )
            if verbose:
                print(
                    "  [final-clean] Requesting answer from saved results "
                    f"(attempt {fallback_attempt})..."
                )
            try:
                clean_kwargs = dict(
                    model=api_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Complete all requested final answers from "
                                "saved measurements. Obey the exact output "
                                "template and return no analysis narrative."
                            ),
                        },
                        {"role": "user", "content": clean_prompt},
                    ],
                    max_tokens=profile['force_answer_max_tokens'],
                    temperature=profile['temperature'],
                )
                _apply_force_answer_controls(clean_kwargs, profile)
                if profile['stream']:
                    clean_kwargs["stream"] = True
                    response = _collect_stream(
                        client.chat.completions.create(**clean_kwargs)
                    )
                else:
                    response = client.chat.completions.create(**clean_kwargs)
                final_message = response.choices[0].message
                final_content = (
                    final_message.content
                    or getattr(final_message, 'reasoning_content', '')
                    or ""
                )
                messages.extend([
                    {"role": "user", "content": clean_prompt},
                    {"role": "assistant", "content": final_content},
                ])
                if _answer_block_complete(question_text, final_content):
                    break
            except Exception as error:
                if verbose:
                    print(
                        f"  [final-clean] Attempt {fallback_attempt} failed: "
                        f"{error}"
                    )

    elapsed = time.time() - t0
    final = _select_final_assistant_text(messages, question_text)

    return {
        "response": final,
        "messages": messages,
        "turns": turn,
        "code_calls": code_calls,
        "elapsed_s": round(elapsed, 1),
    }


def _run_anthropic(question_text, sample_dir, model, max_turns, verbose):
    """Run agent loop using Anthropic-native API."""
    api_key, base_url = get_client_config(model)
    profile = model_execution_profile(model)
    client = Anthropic(api_key=api_key, base_url=base_url)

    gen_max_tokens = profile['generation_max_tokens']

    # Anthropic: system is a separate param, not in messages
    messages = [
        {"role": "user", "content": question_text},
    ]
    # Keep a shadow copy in OpenAI format for the returned conversation
    full_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]

    code_calls = 0
    retries = 0
    t0 = time.time()
    turn = 0

    while turn < max_turns:
        turn += 1
        if verbose:
            print(f"  [turn {turn}/{max_turns}] calling {model}...", end=" ", flush=True)

        try:
            resp = client.messages.create(
                model=profile['api_model'],
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=TOOLS_ANTHROPIC,
                max_tokens=gen_max_tokens,
                temperature=profile['temperature'],
            )
            retries = 0
        except Exception as e:
            retries += 1
            wait = min(10 * (2 ** min(retries - 1, 4)), 120)
            if verbose:
                print(f"API error: {e}, retry #{retries} in {wait}s...")
            if retries > 10:
                if verbose:
                    print("Max retries exceeded, giving up on this problem.")
                break
            time.sleep(wait)
            turn -= 1
            continue

        # Parse Anthropic response: content is a list of blocks
        text_parts = []
        tool_uses = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        assistant_text = "\n".join(text_parts)

        # Build Anthropic-format assistant message (content as list of blocks)
        assistant_content = []
        for block in resp.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})
        full_history.append({"role": "assistant", "content": assistant_text})

        if tool_uses:
            tool_results = []
            for tu in tool_uses:
                if tu.name == "execute_python":
                    code = tu.input.get("code", "")
                    if verbose:
                        first_line = code.strip().split('\n')[0][:60]
                        print(f"exec: {first_line}...")
                    result = execute_python(code, sample_dir)
                    code_calls += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result,
                    })
                    full_history.append({"role": "tool", "content": result})
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": f"Unknown tool: {tu.name}",
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            if verbose:
                print("done (no tool call)")
            break

        if resp.stop_reason == "end_turn" and not tool_uses:
            if verbose:
                print("done (stop)")
            break

    if not any(
            _answer_block_complete(question_text, m.get('content', ''))
            for m in full_history
            if isinstance(m, dict) and m.get('role') == 'assistant'):
        if verbose:
            print("  [final] Requesting structured answer summary...")
        force_msg = _force_answer_prompts(question_text)[0]
        messages.append({"role": "user", "content": force_msg})
        try:
            resp = client.messages.create(
                model=profile['api_model'], system=SYSTEM_PROMPT,
                messages=messages,
                max_tokens=profile['force_answer_max_tokens'],
                temperature=profile['temperature'],
            )
            final_text = ""
            for block in resp.content:
                if block.type == "text":
                    final_text += block.text
            messages.append({"role": "assistant", "content": [{"type": "text", "text": final_text}]})
            full_history.append({"role": "assistant", "content": final_text})
        except Exception as e:
            if verbose:
                print(f"  Force-answer failed: {e}")

    elapsed = time.time() - t0

    final = _select_final_assistant_text(full_history, question_text)

    return {
        "response": final,
        "messages": full_history,
        "turns": turn,
        "code_calls": code_calls,
        "elapsed_s": round(elapsed, 1),
    }


def _executor_protocol_controls():
    return {
        'code_timeout_s': executor_module.CODE_TIMEOUT,
        'max_output_len': executor_module.MAX_OUTPUT_LEN,
        'preamble': executor_module.PREAMBLE,
        'isolation': 'bubblewrap-unshare-all-v1',
    }


def protocol_fingerprint(model, max_turns):
    """Model-aware identity of the current free-form execution protocol."""
    import hashlib
    profile = model_execution_profile(model)
    active_tools = (TOOLS_ANTHROPIC
                    if profile['transport'] == 'anthropic-native'
                    else TOOLS_OPENAI)
    blob = json.dumps({
        'version': FREEFORM_PROTOCOL_VERSION,
        'model_profile': profile,
        'system_prompt': SYSTEM_PROMPT,
        'active_tools': active_tools,
        'force_answer_prompts': [
            _force_answer_prompts(marker)
            for marker in ('emrb-l4-repaired-v1', 'emrb-l2-autocorr-v1',
                           'emrb-l5-verifiable-v5', '')],
        'deepseek_force_answer_addendum': (
            DEEPSEEK_FORCE_ANSWER_ADDENDUM
            if profile['deepseek_v4'] else None
        ),
        'max_turns': max_turns,
        'executor': _executor_protocol_controls(),
        'inline_code_paths': ('native-tool', 'dsml', 'kimi-tool-token',
                              'markdown-code-block'),
        'retry_policy': {'max_retries': 10, 'backoff_cap_s': 120},
    }, sort_keys=True, ensure_ascii=False)
    return {
        'mode': 'free-form',
        'version': FREEFORM_PROTOCOL_VERSION,
        'model': model,
        'provider': profile['provider'],
        'api_model': profile['api_model'],
        'transport': profile['transport'],
        'base_url': profile['base_url'],
        'max_turns': max_turns,
        'clean_final_answer_fallback': profile[
            'clean_final_answer_fallback'
        ],
        'fingerprint': hashlib.md5(blob.encode()).hexdigest(),
    }


def legacy_protocol_fingerprint(model, max_turns):
    """Frozen identity for verified legacy L1/L3 free-form responses.

    Stored conversations establish the prompt and generic force-answer
    family below, but do not preserve a trustworthy provider/API route.
    The protocol therefore records that limitation explicitly instead of
    attributing the responses to today's route.
    """
    import hashlib
    legacy_executor = {
        'code_timeout_s': 60,
        'max_output_len': 15000,
        'preamble': (
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import matplotlib; matplotlib.use('Agg')\n"
        ),
    }
    blob = json.dumps({
        'version': LEGACY_FREEFORM_PROTOCOL_VERSION,
        'requested_model': model,
        'system_prompt': LEGACY_SYSTEM_PROMPT,
        'tools_openai': TOOLS_OPENAI,
        'tools_anthropic': TOOLS_ANTHROPIC,
        'generic_force_answer_prompts': LEGACY_GENERIC_FORCE_PROMPTS,
        'max_turns': max_turns,
        'executor': legacy_executor,
    }, sort_keys=True, ensure_ascii=False)
    return {
        'mode': 'free-form',
        'version': LEGACY_FREEFORM_PROTOCOL_VERSION,
        'model': model,
        'route_status': 'unverified-legacy',
        'max_turns': max_turns,
        'fingerprint': hashlib.md5(blob.encode()).hexdigest(),
    }


def run_problem(question_text, sample_dir, model=None, max_turns=None, verbose=True):
    """Run one EMRB problem through the agent loop."""
    model = model or DEFAULT_MODEL
    max_turns = max_turns or MAX_TURNS

    with isolated_signal_workspace(question_text, sample_dir) as workspace:
        if _is_anthropic_provider(model):
            return _run_anthropic(question_text, workspace, model, max_turns, verbose)
        return _run_openai(question_text, workspace, model, max_turns, verbose)
