# SpeechBot: Voice-Enabled AI Assistant

SpeechBot is a Python-based voice-enabled AI assistant that listens to user queries, processes them using a language model, and responds with engaging, natural-sounding answers. It leverages modern speech recognition and text-to-speech technologies, as well as state-of-the-art NLP models, to create a seamless conversational experience.

---

## Features

- **Voice Interaction**: Listen to user input through a microphone and respond via text-to-speech.
- **Natural Language Processing**: Utilizes the TinyLlama language model for generating context-aware and friendly responses.
- **Dynamic Conversation History**: Maintains a short conversation history to provide context-aware replies.
- **Customizable**: Adjustable speaking rate and volume for the text-to-speech engine.
- **Error Handling**: Graceful handling of various edge cases, such as no speech detected or unclear audio.
- **Multi-Device Compatibility**: Runs on CPU or GPU for flexibility in different environments.

---

## Prerequisites

Ensure the following dependencies are installed before running the project:

1. **Python (>= 3.8)**
2. **Required Libraries**:
   - `speechrecognition`
   - `pyttsx3`
   - `transformers`
   - `torch`
   - `logging`
   - `queue`
   - `threading`

You can install the required Python packages by running:
```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gautamraj8044/Speech-to-speech-bot.git
   cd Speech-to-speech-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the TinyLlama model:
   This project uses the `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model. Ensure the model files are downloaded automatically when the program runs.

4. Verify your microphone and audio output are configured correctly for speech recognition and text-to-speech.

---

## How to Run

1. Run the script:
   ```bash
   python main.py
   ```

2. The assistant will greet you and start listening for input.

3. Interact with the assistant by speaking into your microphone.

4. Say "exit", "quit", or "stop" to end the conversation and close the program.

---

## Project Structure

- **`main.py`**: Main script containing the logic for speech recognition, text-to-speech, and NLP response generation.
- **`requirements.txt`**: List of dependencies for the project.

---

## Configuration

### Text-to-Speech Settings
- Speaking rate: Adjust the rate by modifying `self.engine.setProperty("rate", 150)`.
- Volume: Adjust the volume by modifying `self.engine.setProperty("volume", 0.9)`.

### Model Settings
- Modify the model or tokenizer used by updating the `self.model_name` in the `SpeechBot` class.
- Adjust parameters like `max_new_tokens`, `temperature`, and `top_p` in the `generate_response` method for customized response generation.

---

## Error Handling

- **No speech detected**: If no input is detected within 5 seconds, the bot prompts the user to try again.
- **Unclear audio**: If the input audio cannot be processed, the bot requests the user to speak more clearly.
- **Critical errors**: The bot gracefully shuts down in case of critical errors or keyboard interruptions.

---

## Known Issues

- Background noise may affect the accuracy of speech recognition.
- Long or complex conversations might exceed the model's token limit.

---

## Future Enhancements

- Integration with additional NLP models for more robust responses.
- Multi-language support for speech recognition and response generation.
- Improved handling of background noise and multiple speakers.

---


---

## Contact

For questions or support, please contact:
- **Name**: Gautam Raj
- **Email**: gautamraj8044@gmail.com
- **GitHub**: [https://github.com/gautamraj8044](https://github.com/gautamraj8044)

