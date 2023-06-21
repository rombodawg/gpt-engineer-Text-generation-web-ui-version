import logging
import asyncio
import json
import websockets

logger = logging.getLogger(__name__)

class AI:
    def __init__(self, model, temperature, host="localhost:5005"):
        self.model = model
        self.temperature = temperature
        self.uri = f'ws://{host}/api/v1/stream'  # Define the URI based on the host

    def start(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.next(messages)

    def fsystem(self, msg):
        return {"role": "system", "content": msg}

    def fuser(self, msg):
        return {"role": "user", "content": msg}

    def fassistant(self, msg):
        return {"role": "assistant", "content": msg}

    async def run(self, messages):
        request = {
            'prompt': messages[-1]['content'],  # Considering the last user message as the prompt
            'max_new_tokens': 250,
            'preset': 'None',
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.1,
            'typical_p': 1,
            'epsilon_cutoff': 0, 
            'eta_cutoff': 0,  
            'tfs': 1,
            'top_a': 0,
            'repetition_penalty': 1.18,
            'top_k': 40,
            'min_length': 0,
            'no_repeat_ngram_size': 0,
            'num_beams': 1,
            'penalty_alpha': 0,
            'length_penalty': 1,
            'early_stopping': False,
            'mirostat_mode': 0,
            'mirostat_tau': 5,
            'mirostat_eta': 0.1,
            'seed': -1,
            'add_bos_token': True,
            'truncation_length': 2048,
            'ban_eos_token': False,
            'skip_special_tokens': True,
            'stopping_strings': []
        }
        
        async with websockets.connect(self.uri, ping_interval=None) as websocket:
            await websocket.send(json.dumps(request))

            while True:
                incoming_data = await websocket.recv()
                incoming_data = json.loads(incoming_data)

                if incoming_data['event'] == 'text_stream':
                    return incoming_data['text']
                elif incoming_data['event'] == 'stream_end':
                    return

    async def next(self, messages: list[dict[str, str]], prompt=None):
        if prompt:
            messages = messages + [{"role": "user", "content": prompt}]

        logger.debug(f"Creating a new chat completion: {messages}")
        response = await self.run(messages)  # Call the new run method

        messages = messages + [{"role": "assistant", "content": response}]
        logger.debug(f"Chat completion finished: {messages}")
        return messages
