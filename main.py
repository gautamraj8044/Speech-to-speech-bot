import speech_recognition as sr
import pyttsx3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import time
import warnings
from typing import Optional
import queue
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
logger = logging.getLogger(__name__)

class SpeechBot:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        self.engine.setProperty("volume", 0.9)
        
        self.recognizer = sr.Recognizer()
        self.audio_queue = queue.Queue()
        
        logger.info("Loading TinyLlama model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.conversation_history = []
        logger.info("Model loaded successfully")

    def speak(self, text: str) -> None:
        try:
            logger.info(f"Speaking: {text}")
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            logger.error(f"Error in speak(): {e}")
            self.engine.say("I'm sorry, I encountered an error.")
            self.engine.runAndWait()

    def listen(self) -> Optional[str]:
        with sr.Microphone() as source:
            logger.info("Listening...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                try:
                    text = self.recognizer.recognize_google(audio)
                except sr.RequestError:
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                    except:
                        raise sr.UnknownValueError
                
                logger.info(f"Heard: {text}")
                return text.lower()
                
            except sr.WaitTimeoutError:
                logger.info("No speech detected")
                self.speak("I didn't hear anything. Could you please repeat that?")
                return None
            except sr.UnknownValueError:
                logger.info("Could not understand audio")
                self.speak("I didn't catch that. Could you please speak more clearly?")
                return None
            except Exception as e:
                logger.error(f"Error in listen(): {e}")
                return None

    def generate_response(self, input_text: str) -> str:
        self.conversation_history.append(f"User: {input_text}")
        
        history_context = "\n".join(self.conversation_history[-5:])  
        prompt = f"""<|system|>
        You are a helpful and friendly AI assistant. Keep the conversation light and engaging.
        Previous conversation:
        {history_context}
        <|user|>
        {input_text}
        <|assistant|>"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=100, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_response = response.split("<|assistant|>")[-1].strip()
            
            self.conversation_history.append(f"Assistant: {final_response}")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in generate_response(): {e}")
            return "I apologize, but I'm having trouble generating a response right now."

    def run(self) -> None:
        try:
            self.speak("Hello! I'm ready to chat with you.")
            
            while True:
                user_input = self.listen()
                
                if user_input:
                    if user_input in ["exit", "quit", "goodbye", "stop"]:
                        self.speak("Goodbye! Have a great day!")
                        break
                    
                    response = self.generate_response(user_input)
                    self.speak(response)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            self.speak("Shutting down. Goodbye!")
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
            self.speak("I encountered a critical error and need to shut down.")
        finally:
            # Cleanup
            logger.info("Cleaning up resources...")
            self.engine.stop()

if __name__ == "__main__":
    bot = SpeechBot()
    bot.run()
