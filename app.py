import gradio as gr
import soundfile as sf
from loguru import logger

from asr import *
from llm import *
from tts import *

# load models
chat_model = StreamingGPTProcessor()
tts_model = ThorstenTTS()
asr_model = FasterWhisper()

user_history = []
bot_history = []

def transcribe(audio, history):
    logger.info(f"Received audio of length {len(audio)}")
    sr, audio = audio
    # perform transcription
    transcription, _ = asr_model(audio, sr)
    text = ""
    # add to history (for gradio chat interface)
    history += [[text, None]]

    for segment in transcription:
        text += segment.text + " "
        history[-1][0] = text
        yield history
 
    # keep track of user history for chatbot
    global user_history
    user_history += [text]
    logger.info(f"Completed transcription: {text}")
    yield history


def add_text(text, history):
    history += [[text, None]]
    yield history


def bot(history):
    global bot_history

    transcription = user_history[-1]
    history[-1][1] = ("")
    answer = chat_model(transcription, user_history, bot_history)
    if not isinstance(answer, str):
        for segment in answer:
            history[-1][1] = segment
            yield history
        answer = segment
    else:
        history[-1][1] = answer
    bot_history += [answer]
    
    # perform text-to-speech
    sr, audio_array = tts_model(answer)
    out_file = f"audio_{len(history)}.wav"
    sf.write(out_file, audio_array, sr)
    history += [[None, (out_file,)]]
    yield history

if __name__ == "__main__":
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot([], elem_id="chatbot")

        with gr.Row():
            with gr.Column(scale=0.5):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or record some audio",
                ).style(container=False)
            with gr.Column(scale=0.5):
                audio = gr.Audio(source="microphone", type="numpy", label="Record Audio")
        with gr.Row():
            btn = gr.Button("Submit")

        txt.submit(add_text, [audio, chatbot], [chatbot], queue=False).then(
            bot, chatbot, chatbot
        )

        btn.click(
            transcribe, [audio, chatbot], [chatbot]
        ).then(
            bot, chatbot, chatbot
        )

    demo.queue().launch()
