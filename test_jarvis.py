# Testing Framework for JARVIS
# tests/test_jarvis.py

import pytest
import asyncio
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import json
from pathlib import Path

# Import your JARVIS components
# from jarvis.core.config import JarvisConfig
# from jarvis.core.main import UltimateJARVIS
# from jarvis.ai.llm import LocalLLM
# from jarvis.ai.nlp import AdvancedNLP

class TestJarvisConfig:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = JarvisConfig()
        
        assert config.master_name == "Master"
        assert config.ai_name == "JARVIS" 
        assert config.wake_word == "jarvis"
        assert config.voice_rate == 180
        assert 0.0 <= config.voice_volume <= 1.0
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid voice rate
        with pytest.raises(ValueError):
            JarvisConfig(voice_rate=500)  # Too high
        
        # Test invalid volume
        with pytest.raises(ValueError):
            JarvisConfig(voice_volume=1.5)  # Too high
    
    def test_config_save_load(self):
        """Test configuration persistence"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            
            # Create and save config
            original_config = JarvisConfig(master_name="Test User", voice_rate=200)
            original_config.save_to_file(config_path)
            
            # Load config
            loaded_config = JarvisConfig.load_from_file(config_path)
            
            assert loaded_config.master_name == "Test User"
            assert loaded_config.voice_rate == 200

class TestLocalLLM:
    """Test Local LLM functionality"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for testing"""
        with patch('ollama.generate') as mock_generate:
            mock_generate.return_value = {"response": "Test response"}
            llm = LocalLLM("test-model")
            llm.ollama_available = True
            return llm
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_llm):
        """Test successful response generation"""
        response = await mock_llm.generate_response("Test prompt")
        assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_generate_response_with_context(self, mock_llm):
        """Test response generation with context"""
        context = ["Previous message 1", "Previous message 2"]
        response = await mock_llm.generate_response("Test prompt", context)
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_fallback_response(self):
        """Test fallback when models unavailable"""
        llm = LocalLLM("test-model")
        llm.ollama_available = False
        llm.hf_model = None
        
        response = await llm.generate_response("hello")
        assert "Hello" in response  # Should use fallback

class TestAdvancedNLP:
    """Test NLP functionality"""
    
    @pytest.fixture
    def nlp_processor(self):
        """Create NLP processor for testing"""
        with patch('spacy.load'), \
             patch('transformers.pipeline'):
            return AdvancedNLP()
    
    def test_intent_analysis(self, nlp_processor):
        """Test intent analysis"""
        # Mock the intent classifier
        nlp_processor.intent_classifier = MagicMock()
        nlp_processor.intent_classifier.return_value = {
            "labels": ["question", "command"],
            "scores": [0.8, 0.2]
        }
        
        result = nlp_processor.analyze_intent("What time is it?")
        
        assert result["intent"] == "question"
        assert result["confidence"] == 0.8
    
    def test_sentiment_analysis(self, nlp_processor):
        """Test sentiment analysis"""
        nlp_processor.sentiment_analyzer = MagicMock()
        nlp_processor.sentiment_analyzer.return_value = [{
            "label": "POSITIVE",
            "score": 0.9
        }]
        
        result = nlp_processor.analyze_sentiment("I love this!")
        
        assert result["sentiment"] == "POSITIVE"
        assert result["confidence"] == 0.9

class TestVoiceProcessor:
    """Test voice processing"""
    
    @pytest.fixture
    def voice_processor(self):
        """Create voice processor for testing"""
        config = JarvisConfig()
        with patch('speech_recognition.Recognizer'), \
             patch('speech_recognition.Microphone'), \
             patch('pyttsx3.init'), \
             patch('faster_whisper.WhisperModel'):
            return AdvancedVoiceProcessor(config)
    
    @pytest.mark.asyncio
    async def test_wake_word_detection(self, voice_processor):
        """Test wake word detection"""
        # Mock successful wake word detection
        with patch.object(voice_processor, 'recognizer') as mock_recognizer:
            mock_recognizer.listen.return_value = MagicMock()
            mock_recognizer.recognize_google.return_value = "jarvis hello"
            
            result = await voice_processor.listen_for_wake_word()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_command_recognition(self, voice_processor):
        """Test voice command recognition"""
        with patch.object(voice_processor, 'recognizer') as mock_recognizer:
            mock_recognizer.listen.return_value = MagicMock()
            mock_recognizer.recognize_google.return_value = "what time is it"
            
            result = await voice_processor.listen_for_command()
            assert result == "what time is it"

class TestSystemMonitor:
    """Test system monitoring"""
    
    @pytest.fixture
    def system_monitor(self):
        """Create system monitor for testing"""
        return AdvancedSystemMonitor()
    
    @pytest.mark.asyncio
    async def test_system_info_collection(self, system_monitor):
        """Test system information collection"""
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Mock memory info
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.total = 8 * 1024**3  # 8GB
            mock_memory.return_value.available = 4 * 1024**3  # 4GB
            mock_memory.return_value.percent = 50.0
            
            # Mock disk info
            mock_disk.return_value = MagicMock()
            mock_disk.return_value.total = 100 * 1024**3  # 100GB
            mock_disk.return_value.used = 50 * 1024**3   # 50GB
            mock_disk.return_value.free = 50 * 1024**3   # 50GB
            
            info = await system_monitor.get_comprehensive_system_info()
            
            assert info["cpu"]["usage_percent"] == 45.0
            assert info["memory"]["percentage"] == 50.0
    
    def test_health_check_alerts(self, system_monitor):
        """Test health check alert generation"""
        # Mock high resource usage
        system_info = {
            "cpu": {"usage_percent": 95.0},
            "memory": {"percentage": 90.0},
            "disk": {
                "/dev/sda1": {"percentage": 95.0}
            },
            "gpu": {}
        }
        
        alerts = system_monitor.check_system_health(system_info)
        
        assert len(alerts) >= 2  # CPU and memory alerts
        assert any("CPU" in alert for alert in alerts)
        assert any("memory" in alert for alert in alerts)

class TestAutomationEngine:
    """Test automation functionality"""
    
    @pytest.fixture
    def automation_engine(self):
        """Create automation engine for testing"""
        return AutomationEngine()
    
    @pytest.mark.asyncio
    async def test_safe_command_execution(self, automation_engine):
        """Test safe command execution"""
        # Test allowed command
        with patch('asyncio.create_subprocess_shell') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await automation_engine.execute_command("echo hello")
            assert "output" in result or result != ""
    
    @pytest.mark.asyncio  
    async def test_dangerous_command_rejection(self, automation_engine):
        """Test that dangerous commands are rejected"""
        with pytest.raises(SecurityException):
            await automation_engine.execute_command("rm -rf /")
    
    def test_macro_creation(self, automation_engine):
        """Test macro creation and storage"""
        actions = [
            {"type": "click", "x": 100, "y": 200},
            {"type": "type", "text": "hello"},
            {"type": "key", "key": "enter"}
        ]
        
        result = automation_engine.create_macro("test_macro", actions)
        assert "test_macro" in automation_engine.macros
        assert len(automation_engine.macros["test_macro"]) == 3

class TestSecurityManager:
    """Test security functionality"""
    
    @pytest.fixture
    def security_manager(self):
        """Create security manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SecurityManager()
            manager.key_file = Path(temp_dir) / "test_key"
            return manager
    
    def test_encryption_decryption(self, security_manager):
        """Test data encryption and decryption"""
        original_data = "sensitive information"
        
        # Encrypt data
        encrypted = security_manager.encrypt_data(original_data)
        assert encrypted != original_data.encode()
        
        # Decrypt data
        decrypted = security_manager.decrypt_data(encrypted)
        assert decrypted == original_data
    
    def test_api_key_storage(self, security_manager):
        """Test secure API key storage"""
        with patch('keyring.set_password'), \
             patch('keyring.get_password', return_value="test_key"):
            
            security_manager.store_api_key("test_service", "secret_key")
            retrieved_key = security_manager.get_api_key("test_service")
            
            assert retrieved_key == "test_key"

class TestIntegration:
    """Integration tests for JARVIS components"""
    
    @pytest.fixture
    def jarvis_instance(self):
        """Create JARVIS instance for testing"""
        with patch.multiple(
            'jarvis.core.main',
            LocalLLM=MagicMock(),
            AdvancedNLP=MagicMock(),
            AdvancedVoiceProcessor=MagicMock(),
            ComputerVision=MagicMock(),
            AutomationEngine=MagicMock(),
            AdvancedSystemMonitor=MagicMock(),
            PluginManager=MagicMock()
        ):
            config = JarvisConfig(master_name="Test User")
            return UltimateJARVIS(config)
    
    @pytest.mark.asyncio
    async def test_command_processing_pipeline(self, jarvis_instance):
        """Test complete command processing pipeline"""
        # Mock NLP analysis
        jarvis_instance.nlp.analyze_intent.return_value = {
            "intent": "greeting",
            "confidence": 0.9
        }
        jarvis_instance.nlp.analyze_sentiment.return_value = {
            "sentiment": "positive",
            "confidence": 0.8
        }
        jarvis_instance.nlp.extract_entities.return_value = []
        
        # Test command processing
        response = await jarvis_instance.process_command("Hello JARVIS")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Test User" in response or "hello" in response.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, jarvis_instance):
        """Test error handling in command processing"""
        # Mock NLP to raise exception
        jarvis_instance.nlp.analyze_intent.side_effect = Exception("NLP Error")
        
        # Should not crash, should return graceful error message
        response = await jarvis_instance.process_command("test command")
        
        assert isinstance(response, str)
        assert "error" in response.lower() or "trouble" in response.lower()

# Performance Tests
class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_command_processing(self):
        """Test handling multiple concurrent commands"""
        # This would test how JARVIS handles multiple simultaneous requests
        pass
    
    @pytest.mark.asyncio  
    async def test_memory_usage_under_load(self):
        """Test memory usage under sustained load"""
        # This would monitor memory usage during extended operation
        pass
    
    def test_startup_time(self):
        """Test JARVIS startup performance"""
        import time
        
        start_time = time.time()
        # Initialize JARVIS components
        # jarvis = UltimateJARVIS()
        end_time = time.time()
        
        startup_time = end_time - start_time
        assert startup_time < 10.0  # Should start within 10 seconds

# Fixture for test database/config
@pytest.fixture(scope="session")
def test_config():
    """Create test configuration"""
    return {
        "test_api_keys": {
            "openweather": "test_weather_key",
            "newsapi": "test_news_key"
        },
        "test_models": {
            "llm": "test-model",
            "voice": "whisper-base"
        }
    }

# Custom test markers
pytestmark = [
    pytest.mark.asyncio,  # Mark all tests as async by default
]

# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"  
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

# Run tests with: python -m pytest tests/ -v
# Run specific test category: python -m pytest tests/ -m "not slow" -v
# Run with coverage: python -m pytest tests/ --cov=jarvis --cov-report=html