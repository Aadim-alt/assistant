# Suggested refactored project structure

# jarvis/
# ├── core/
# │   ├── __init__.py
# │   ├── config.py          # JarvisConfig class
# │   ├── security.py        # SecurityManager class  
# │   └── main.py           # UltimateJARVIS main class
# ├── ai/
# │   ├── __init__.py
# │   ├── nlp.py            # AdvancedNLP class
# │   ├── llm.py            # LocalLLM class
# │   └── voice.py          # AdvancedVoiceProcessor class
# ├── vision/
# │   ├── __init__.py
# │   └── computer_vision.py # ComputerVision class
# ├── automation/
# │   ├── __init__.py
# │   └── engine.py         # AutomationEngine class
# ├── monitoring/
# │   ├── __init__.py
# │   └── system_monitor.py # AdvancedSystemMonitor class
# ├── gui/
# │   ├── __init__.py
# │   └── interface.py      # ModernGUI class
# ├── api/
# │   ├── __init__.py
# │   └── web_api.py        # WebAPI class
# ├── plugins/
# │   ├── __init__.py
# │   ├── manager.py        # PluginManager class
# │   └── examples/         # Sample plugins
# ├── utils/
# │   ├── __init__.py
# │   └── helpers.py        # Utility functions
# ├── requirements.txt
# ├── setup.py
# └── main.py              # Entry point

# Example: core/config.py
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

class AIModel(Enum):
    """Available AI models"""
    OLLAMA_LLAMA2 = "llama2"
    OLLAMA_MISTRAL = "mistral"
    OLLAMA_CODELLAMA = "codellama"
    HUGGINGFACE_GPTJ = "EleutherAI/gpt-j-6B"
    HUGGINGFACE_GPTNEOX = "EleutherAI/gpt-neox-20b"
    LOCAL_WHISPER = "whisper-base"

@dataclass
class JarvisConfig:
    """Configuration class for JARVIS"""
    master_name: str = "Master"
    ai_name: str = "JARVIS"
    wake_word: str = "jarvis"
    voice_model: str = "whisper-base"
    llm_model: str = "llama2"
    voice_rate: int = 180
    voice_volume: float = 0.9
    language: str = "en"
    theme: str = "dark"
    api_timeout: int = 30
    max_context_length: int = 4096
    enable_vision: bool = True
    enable_automation: bool = True
    enable_learning: bool = True
    debug_mode: bool = False

    @classmethod
    def load_from_file(cls, config_path: Path) -> 'JarvisConfig':
        """Load configuration from JSON file"""
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                return cls(**config_dict)
        return cls()

    def save_to_file(self, config_path: Path):
        """Save configuration to JSON file"""
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

# Example: ai/llm.py  
import asyncio
import logging
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class LocalLLM:
    """Local Large Language Model integration"""
    
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.ollama_available = False
        self.hf_model = None
        self.hf_tokenizer = None
        self._context_cache = {}
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM with better error handling"""
        try:
            self._try_ollama()
        except Exception as e:
            logger.warning(f"Ollama initialization failed: {e}")
            try:
                self._initialize_huggingface_model()
            except Exception as hf_error:
                logger.error(f"All LLM initialization failed: {hf_error}")
                raise RuntimeError("No LLM backend available")
    
    def _try_ollama(self):
        """Try to initialize Ollama"""
        import ollama
        # Test if Ollama is running
        try:
            ollama.list()  # Test connection
            self.ollama_available = True
            logger.info(f"Ollama connected successfully")
        except Exception:
            raise ConnectionError("Ollama not running")
    
    async def generate_response(self, prompt: str, context: List[str] = None, max_tokens: int = 512) -> str:
        """Generate response with better context management"""
        try:
            # Build context-aware prompt with caching
            cache_key = hash(prompt + str(context))
            if cache_key in self._context_cache:
                logger.debug("Using cached context")
            
            if self.ollama_available:
                return await self._ollama_generate(prompt, context, max_tokens)
            elif self.hf_model:
                return await self._huggingface_generate(prompt, max_tokens)
            else:
                return self._fallback_response(prompt)
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I apologize, but I'm having trouble processing that request."
    
    async def _ollama_generate(self, prompt: str, context: List[str] = None, max_tokens: int = 512) -> str:
        """Generate response using Ollama with better configuration"""
        try:
            import ollama
            
            # Improved context building
            if context:
                # Only use recent context to avoid token limits
                recent_context = context[-3:] if len(context) > 3 else context
                full_prompt = "\n".join(recent_context) + "\nUser: " + prompt + "\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"
            
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_name,
                prompt=full_prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": max_tokens,
                    "stop": ["User:", "\n\n"]
                }
            )
            return response["response"].strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise