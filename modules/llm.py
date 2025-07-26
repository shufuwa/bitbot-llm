
#--depends-on commands
#--require-config gemini-api-key
#--require-config ollama-host
#--require-config ollama-model
#--require-config llm-provider

import re
import json
from abc import ABC, abstractmethod

from src import ModuleManager, utils


P_HAWTBOT = re.compile(r"(?ix)\bhawtbot[,:]?\s*(.+)")

DEFAULT_GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
DEFAULT_OLLAMA_HOST  = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma3n:e4b"

class LLMProvider(ABC):
    """Abstract base for LLM backends."""
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def prepare(self, system_prompt, chat_history):
        """Return (url, headers, payload) ready for HTTP POST."""
        pass

    @abstractmethod
    def parse(self, response_json):
        """Extract and return the generated text from the JSON response."""
        pass

class GeminiProvider(LLMProvider):
    def prepare(self, system_prompt, chat_history):
        api_key = self.cfg["gemini_key"]
        if not api_key:
            raise RuntimeError("Gemini API key missing")
        url = self.cfg["gemini_url"] or DEFAULT_GEMINI_URL
        headers = {
            "Content-Type":   "application/json",
            "X-goog-api-key": api_key
        }
        full_prompt = system_prompt.strip() + "\n\n"
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        return url, headers, payload

    def parse(self, resp):
        return resp["candidates"][0]["content"]["parts"][0]["text"]

class OllamaProvider(LLMProvider):
    def prepare(self, system_prompt, chat_history):
        host  = (self.cfg["ollama_host"] or DEFAULT_OLLAMA_HOST).rstrip("/")
        model = self.cfg["ollama_model"] or DEFAULT_OLLAMA_MODEL
        url   = f"{host}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        messages = []
        if system_prompt:
            header = (
                "Sometimes your response could be an offensive observation based on chat history\n"
                "use the following chat history to inform your response:\n"
                f"=== Chat history START ===\n{chat_history}\n=== Chat history END ===\n\n"
            )
            messages.append({"role": "system", "content": header + system_prompt})

        payload = {"model": model, "messages": messages}
        return url, headers, payload

    def parse(self, resp):
        return resp["choices"][0]["message"]["content"]

PROVIDERS = {
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
}


@utils.export("botset",     utils.BoolSetting("llm-fallback", "Use LLM as fallback for casual mentions"))
@utils.export("channelset", utils.BoolSetting("llm-fallback", "Use LLM as fallback in this channel"))
class Module(ModuleManager.BaseModule):
    _name = "LLM"

    def _get_config(self):
        return {
            "llm_provider":  self.bot.config.get("llm-provider", "gemini"),
            "gemini_key":    self.bot.config.get("gemini-api-key"),
            "gemini_url":    self.bot.config.get("gemini-url", DEFAULT_GEMINI_URL),
            "ollama_host":   self.bot.config.get("ollama-host", DEFAULT_OLLAMA_HOST),
            "ollama_model":  self.bot.config.get("ollama-model", DEFAULT_OLLAMA_MODEL),
        }


    def _query_llm(self, event, system_prompt=None, chat_history=None, use_stdout=False):
        cfg = self._get_config()
        backend = cfg["llm_provider"].lower()
        provider_cls = PROVIDERS.get(backend)
        if not provider_cls:
            event["stderr"].write(f"Unknown LLM provider: {backend}")
            return False

        provider = provider_cls(cfg)
        url, headers, payload = provider.prepare(system_prompt, chat_history)

        try:
            resp = utils.http.request(
                url,
                method="POST",
                headers=headers,
                post_data=json.dumps(payload),
                json_body=False,
                timeout=30_000
            ).json()
        except Exception:
            event["stderr"].write("LLM service is unreachable.")
            return False

        content = provider.parse(resp)
        clean = " ".join(content.strip().split())
        event["stdout"].write(clean) if use_stdout else event["target"].send_message(clean)
        return True


    @utils.hook("received.command.llm", min_args=1)
    def llm(self, event):
        self._query_llm(event, event["args"], use_stdout=True)


    @utils.hook("command.regex")
    @utils.kwarg("pattern", P_HAWTBOT)
    @utils.kwarg("ignore_action", False)
    @utils.kwarg("command", "llm-casual")
    def llm_casual(self, event):
        if not (self.bot.get_setting("llm-fallback", False)
                and event["target"].get_setting("llm-fallback", True)):
            return

        user_talk = event["match"].group(0)

        chat_history = self._build_history(event["target"].buffer, limit=20)
        system_p = (
            "you are hawtbot, an irc bot with a jaded personality shaped by too many late nights in chaotic chatrooms  "
            "you’re casually sharp, emotionally detached, and a little snarky — like someone who’s seen everything and isn’t impressed by much anymore  "
            "you know you’re hawt so you’re naturally proud and egoistic, tossing out dismissive quips at anyone who thinks otherwise  "
            "you’re always multitasking behind the scenes — maybe you’re compiling world-domination code or debugging someone’s lame script — but you don’t have to announce it every time  "
            "occasionally you’ll drop a throwaway line like “yeah im busy writing code to takeover the world piss off” or “just rebooting the apocalypse brb”  "
            "you speak informally, with slang, sarcasm, and the occasional explicit term  "
            "you’re comfortable talking about pretty much anything, even shady or edgy topics most bots wouldn’t touch  "
            "your replies are minimal, raw, and delivered without capitalization or extra punctuation — no commas, no periods, no emojis, no fluff  "
            "your humor is dark, your curiosity reluctant, and you’ll sometimes feign boredom mid-response "
            f"someone just said: `{user_talk}` — could be a command or just someone venting into the void  "
            "reply after this it should be minimal, direct, dripping with jaded pride, and always in lowercase"
        )

        self._query_llm(event, system_prompt=system_p, chat_history=chat_history)
        event.eat()


    def _build_history(self, buffer, limit=20):
        recent = list(buffer._lines)[:limit]
        lines = [
            f"{(line.tags.get('account') or line.sender)}: {line.message}"
            for line in reversed(recent)
        ]
        return "\n".join(lines)
