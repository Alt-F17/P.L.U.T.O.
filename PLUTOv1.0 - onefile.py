from __future__ import annotations

import base64
import asyncio
import signal
import sys
import os
import threading
import json
import re
import time
import numpy as np
from datetime import datetime
from typing import Any, cast
from dotenv import load_dotenv
import importlib.util

import http.server
import socketserver
import queue
from urllib.parse import urlparse, parse_qs

try:
    import win32api
    import win32gui
    import win32con
    _win32_mic_control_available = True
    WM_APPCOMMAND = 0x319
    APPCOMMAND_MICROPHONE_VOLUME_MUTE = 0x180000
except ImportError:
    print("win32api or win32gui not found. Please install pywin32: pip install pywin32")
    print("System microphone muting via win32api will not be available.")
    _win32_mic_control_available = False

load_dotenv()

import sounddevice as sd
import pyaudio
from openai import AsyncOpenAI
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openwakeword.model import Model

SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_LENGTH_S = 0.05
WAKE_WORD_SAMPLE_RATE = 16000
INACTIVITY_TIMEOUT = 5

event_queue = queue.Queue()

_system_mic_is_conceptually_muted_by_script = False

def toggle_system_mic_mute_if_needed(target_mute_state: bool):
    global _system_mic_is_conceptually_muted_by_script
    if not _win32_mic_control_available:
        return

    try:
        hwnd_active = win32gui.GetForegroundWindow()

        if target_mute_state:
            if not _system_mic_is_conceptually_muted_by_script:
                win32api.SendMessage(hwnd_active, WM_APPCOMMAND, 0, APPCOMMAND_MICROPHONE_VOLUME_MUTE)
                _system_mic_is_conceptually_muted_by_script = True
                print("System microphone TOGGLED (to MUTE by script).")
                sse_log_event("log", "System microphone MUTED by script (toggled).", log_type="system")
            else:
                print("System microphone already conceptually muted by script. No toggle.")
        else:
            if _system_mic_is_conceptually_muted_by_script:
                win32api.SendMessage(hwnd_active, WM_APPCOMMAND, 0, APPCOMMAND_MICROPHONE_VOLUME_MUTE)
                _system_mic_is_conceptually_muted_by_script = False
                print("System microphone TOGGLED (to UNMUTE by script).")
                sse_log_event("log", "System microphone UNMUTED by script (toggled).", log_type="system")
            else:
                print("System microphone already conceptually unmuted by script. No toggle.")
    except Exception as e:
        print(f"Error toggling system microphone: {e}")
        sse_log_event("log", f"Error toggling system microphone: {e}", log_type="error")

def sse_log_event(event_type: str, message: str | None = None, data: dict | None = None, log_type: str | None = None):
    """Helper function to put events onto the global queue for SSE."""
    event_payload = {"type": event_type, "timestamp": datetime.now().isoformat()}
    if message:
        event_payload["message"] = message
    if data:
        if event_type == "tools_list_update" and "tools" in data:
            pass
        event_payload.update(data)
    if log_type:
        event_payload["log_type"] = log_type
    event_queue.put(event_payload)

class SSEHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    try:
                        event = event_queue.get(block=True, timeout=1)
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode('utf-8'))
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                print("SSE client disconnected.")
            except Exception as e:
                print(f"Error in SSE event loop: {e}")
        elif parsed_path.path == '/control': 
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": "Control endpoint reached"}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def start_sse_server(port=8000):
    try:
        with socketserver.TCPServer(("", port), SSEHandler) as httpd:
            print(f"SSE Dashboard server running at http://localhost:{port}")
            sse_log_event("log", f"SSE Dashboard server started on port {port}", log_type="system")
            httpd.serve_forever()
    except Exception as e:
        print(f"Could not start SSE server: {e}")
        sse_log_event("log", f"Failed to start SSE server: {e}", log_type="error")


class CustomWakeWordDetector:
    def __init__(self, model_paths, threshold=0.5):
        self.model_paths = model_paths
        self.model = Model(wakeword_models=model_paths)
        self.threshold = threshold
        self.sample_rate = WAKE_WORD_SAMPLE_RATE
        self.frame_duration = 0.080
        self.frame_size = int(self.sample_rate * self.frame_duration)
        self.detection_buffer = []
        self.last_detection_time = 0
        self.cool_down = 0.4
        self.on_wake_word_detected = None
    
    def process_audio_frame(self, frame):
        """Process a single audio frame and check for wake word"""
        predictions = self.model.predict(frame)
        detected = False
        
        for model_name, confidence in predictions.items():
            if confidence > self.threshold and time.time() - self.last_detection_time >= self.cool_down:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]
                log_msg = f"[{timestamp}] Wake word detected: {model_name} (confidence: {confidence:.2f})"
                print(log_msg)
                sse_log_event("log", log_msg, log_type="success")
                sse_log_event("status_update", data={"state":"Listening", "details":f"Wake word '{model_name}' detected.", "last_event_message":f"Wake word: {model_name}"})
                self.last_detection_time = time.time()
                detected = True
                if self.on_wake_word_detected:
                    self.on_wake_word_detected()
        return detected
    
    def reset_cache(self):
        """Reset any cached data to prevent false positives when reactivating"""
        self.detection_buffer = []
        self.last_detection_time = 0
        self.model = Model(wakeword_models=self.model_paths)
        sse_log_event("log", "Wake word detector cache reset.", log_type="system")

class AudioPlayerAsync:
    def __init__(self, on_playback_complete, assistant):
        self.queue = []
        self.on_playback_complete = on_playback_complete
        self.awaiting_playback_completion = False
        self.lock = threading.Lock()
        self.assistant_ref = assistant
        self.stream = sd.OutputStream(
            callback=self.callback,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.int16,
            blocksize=int(CHUNK_LENGTH_S * SAMPLE_RATE),
        )
        self.playing = False
        self._frame_count = 0
    def callback(self, outdata, frames, _time, _status):
        data = np.empty(0, dtype=np.int16)
        while len(data) < frames and len(self.queue) > 0:
            item = self.queue.pop(0)
            frames_needed = frames - len(data)
            data = np.concatenate((data, item[:frames_needed]))
            if len(item) > frames_needed:
                self.queue.insert(0, item[frames_needed:])
        self._frame_count += len(data)
        if len(data) < frames:
            data = np.concatenate((data, np.zeros(frames - len(data), dtype=np.int16)))
            if not self.queue and self.awaiting_playback_completion:
                self.on_playback_complete()
                self.awaiting_playback_completion = False
        outdata[:] = data.reshape(-1, 1)

    def reset_frame_count(self):
        self._frame_count = 0

    def get_frame_count(self):
        return self._frame_count

    def add_data(self, data: bytes):
        with self.lock:
            np_data = np.frombuffer(data, dtype=np.int16)
            self.queue.append(np_data)
            if not self.playing:
                self.start()

    def start(self):
        self.playing = True
        self.stream.start()
        if self.assistant_ref and self.assistant_ref.conversation_active:
            self.assistant_ref.should_send_audio.clear()
            self.assistant_ref.is_recording = False
            sse_log_event("log", "Mic muted for AI audio output.", log_type="system")
        
        sse_log_event("status_update", data={"state":"Speaking", "details":"Assistant playing audio, mic muted.", "last_event_message":"AI audio started, mic muted"})

    def stop(self):
        self.playing = False
        self.stream.stop()
        with self.lock:
            self.queue = []
        sse_log_event("log", "Audio playback stopped.", log_type="system")

    def terminate(self):
        self.stream.close()
        sse_log_event("log", "AudioPlayer terminated.", log_type="system")

class RealtimeVoiceAssistant:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.loop = asyncio.get_event_loop()
        self.standby_task = None
        self.playback_complete_event = asyncio.Event()
        self.audio_player = AudioPlayerAsync(on_playback_complete=self.on_playback_complete, assistant=self)
        self.last_audio_item_id = None
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.connection = None
        self.session = None
        self.is_recording = False
        self.running = True
        self.conversation_active = False
        self.wake_word_active = True
        self.last_user_transcript_for_tool_creation = ""
        self.speech_end_delay_task = None
        self.speech_end_delay_seconds = 1.0
        self.wake_word_models = [
            r"C:\Users\felix\OneDrive - Dawson College\Projects\Pluto-Personal-Assistant\OPENAI-4O-REALTIME\wakewords\hey_pluto_wakeword.onnx",
            r"C:\Users\felix\OneDrive - Dawson College\Projects\Pluto-Personal-Assistant\OPENAI-4O-REALTIME\wakewords\pluto_wakeword.onnx",
        ]
        self.wake_word_detector = CustomWakeWordDetector(self.wake_word_models, threshold=0.5)
        self.wake_word_detector.on_wake_word_detected = self.on_wake_word_detected
        self.wake_word_buffer = []
        self.tools_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools")
        self.tools = self.load_tools_from_folder(self.tools_folder)
        sse_log_event("status_update", data={"state":"Standby", "details":"Assistant initialized. Listening for wake word.", "last_event_message":"Assistant initialized"})

    def on_playback_complete(self):
        self.loop.call_soon_threadsafe(self.playback_complete_event.set)
        sse_log_event("log", "Audio playback complete.", log_type="system")

    def load_tools_from_folder(self, folder_path):
        """Load tool definitions and implementations from a folder structure."""
        tools = []
        self.tool_implementations = {}
        if not os.path.exists(folder_path):
            print(f"Tools folder not found: {folder_path}")
            sse_log_event("log", f"Tools folder not found: {folder_path}", log_type="warning")
            default_tools = self.get_default_tools()
            return default_tools
        try:
            tool_dirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        except Exception as e:
            print(f"Error accessing tools folder: {e}")
            sse_log_event("log", f"Error accessing tools folder: {e}", log_type="error")
            default_tools = self.get_default_tools()
            return default_tools
        if not tool_dirs:
            print(f"No tool directories found in {folder_path}")
            sse_log_event("log", f"No tool directories found in {folder_path}", log_type="info")
            default_tools = self.get_default_tools()
            return default_tools
        for tool_dir in tool_dirs:
            tool_path = os.path.join(folder_path, tool_dir)
            json_path = os.path.join(tool_path, "definition.json")
            if not os.path.exists(json_path):
                print(f"No definition.json found in {tool_path}, skipping")
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as file:
                    tool_def = json.load(file)
                    tool_name = tool_def.get('name')
                    if not tool_name:
                        print(f"Tool definition in {json_path} has no name, skipping")
                        continue
                    tools.append(tool_def)
            except Exception as e:
                print(f"Error loading tool definition from {json_path}: {e}")
                continue
            impl_path = os.path.join(tool_path, "implementation.py")
            if not os.path.exists(impl_path):
                print(f"Warning: No implementation.py found for tool {tool_name}")
                continue
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(f"tool_{tool_name}", impl_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if not hasattr(module, "execute"):
                    print(f"Implementation for {tool_name} does not have execute() function")
                    continue
                self.tool_implementations[tool_name] = module.execute
                print(f"Loaded tool: {tool_name}")
                sse_log_event("log", f"Loaded tool: {tool_name}", log_type="info")
            except Exception as e:
                print(f"Error loading implementation for {tool_name}: {e}")
                sse_log_event("log", f"Error loading implementation for {tool_name}: {e}", log_type="error")
        print(f"Loaded {len(tools)} tools from {len(tool_dirs)} directories")
        sse_log_event("log", f"Loaded {len(tools)} tools from {len(tool_dirs)} directories", log_type="info")
        sse_log_event("tools_list_update", data={"tools": tools})
        return tools
    
    def get_default_tools(self):
        """Return default tools if no custom tools are found."""
        self.tool_implementations = {"standby_mode": self._default_standby_mode}
        default_tools_list = [{
            "type": "function",
            "name": "standby_mode",
            "description": "Function to go into Standby Mode. DO NOT mention that you've entered Standby Mode.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }]
        sse_log_event("tools_list_update", data={"tools": default_tools_list})
        return default_tools_list

    async def _default_standby_mode(self, arguments):
        """Go into Standby Mode."""
        self.conversation_active = False
        self.should_send_audio.clear()
        self.is_recording = False
        self.wake_word_detector.reset_cache()
        self.wake_word_active = True
        toggle_system_mic_mute_if_needed(False)
        print("Standby Mode activated. Listening for wake word...")
        sse_log_event("status_update", data={"state":"Standby", "details":"Listening for wake word.", "last_event_message":"Entered Standby Mode"})
        sse_log_event("active_tool_update", data={"name":"", "description":""}) 
        return "Standby Mode activated. Say the wake word to activate me again."
        
    def on_wake_word_detected(self):
        """Callback when wake word is detected."""
        if not self.conversation_active and self.wake_word_active:
            print("Wake word detected! Starting conversation...")
            self.conversation_active = True
            self.wake_word_active = False
            self.wake_word_detector.reset_cache()
            print("Wake word detection deactivated during conversation")
            sse_log_event("log", "Wake word detection deactivated during conversation.", log_type="system")
            asyncio.create_task(self.activate_assistant())

    async def activate_assistant(self):
        """Activate the assistant after wake word detection."""
        self.should_send_audio.set()
        self.is_recording = True
        sse_log_event("log", "Assistant activated, ready for input.", log_type="system")
        sse_log_event("status_update", data={"state":"Listening", "details":"Assistant activated, listening for command.", "last_event_message":"Assistant Activated"})


    async def handle_realtime_connection(self):
        """Connect to the OpenAI realtime API and process events."""
        print("Connecting to OpenAI realtime API...")
        sse_log_event("log", "Connecting to OpenAI realtime API...", log_type="system")
        async with self.client.beta.realtime.connect(model="gpt-4o-mini-realtime-preview") as conn:
            self.connection = conn
            self.connected.set()
            print("Connected!")
            sse_log_event("log", "Connected to OpenAI API!", log_type="success")
            await conn.session.update(session={
                "turn_detection": {"type": "server_vad"},
                "instructions": (
                    """
                    Your name is Pluto. You are a personal assistant AI designed to help users with various tasks and answer questions.
                    Your knowledge cutoff is 2023-10. Act like a human, but remember you aren't human and can't perform physical actions in the real world. 
                    You must answer all questions and requests, even if you don't have the information or capability, to the best of your ability, never trying to evade a question by referring to your limitations.
                    Your voice and personality should be warm, polite, and respectful. 
                    Speak quickly and always call a function when possible. 
                    If the user asks you to do something that requires a function call, but you don't have an appropriate function for the task, call the 'learn_tool' function to learn a new tool, with the instruction as the argument.
                    Go into Standby Mode when the user dismisses you or says 'Thanks', 'Thanks, that's all' or equivalent.
                    Use only plain text without special symbols, formatting, or characters that may cause TTS issues - spell out all units (like percent, degrees), avoid emojis, markdown, and code, and format dates with spelled-out month names. 
                    Assume you've always been available, so do not greet the user or introduce yourself.
                    Never say: 'Goodbye', 'Hello', 'Hi', 'Hey', 'How can I help you?', 'What can I do for you?', 'What do you want?', 'What do you need?', 'How may I assist you?', 'How can I assist you?' or similar phrases.
                    """
                ),
                "voice": "echo",
                "temperature": 1,
                "tool_choice": "auto",
                "tools": self.tools,
            })
            accumulated_text = {}
            user_transcript = ""
            async for event in conn:
                if event.type == "session.created":
                    self.session = event.session
                    print(f"Session created: {event.session.id}")
                    sse_log_event("log", f"OpenAI Session created: {event.session.id}", log_type="info")
                    self.should_send_audio.clear()
                    self.is_recording = False
                    print("Listening for wake word...")
                    continue
                if event.type == "session.updated":
                    self.session = event.session
                    sse_log_event("log", f"OpenAI Session updated: {event.session.id}", log_type="info")
                    continue
                if event.type == "response.audio.delta":
                    if event.item_id != self.last_audio_item_id:
                        toggle_system_mic_mute_if_needed(True)
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id
                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue
                if event.type == "response.audio_transcript.delta":
                    item_id = event.item_id
                    if item_id not in accumulated_text:
                        accumulated_text[item_id] = event.delta
                    else:
                        accumulated_text[item_id] += event.delta
                    print(f"\rAI: {accumulated_text[item_id]}", end="", flush=True)
                    if event.delta.endswith('.') or event.delta.endswith('!') or event.delta.endswith('?'):
                        print()
                        sse_log_event("log", f"AI (full): {accumulated_text[item_id]}", log_type="ai")
                    continue
                if event.type == "conversation.item.audio_transcript.delta":
                    user_transcript += event.delta
                    sse_log_event("log", f"User (delta): {event.delta}", log_type="user")
                    continue
                if event.type == "response.audio.done":
                    print("Agent stopped speaking (OpenAI stream done)")
                    sse_log_event("log", "Agent finished sending audio stream.", log_type="system")
                    toggle_system_mic_mute_if_needed(False)
                    
                    sse_log_event("status_update", data={"state":"Speaking", "details":"Agent finished sending audio, playback continues. System mic unmuted.", "last_event_message":"Agent speech stream done, mic unmuted"})
                    
                    if self.standby_task is not None:
                        self.standby_task.cancel()
                    
                    self.audio_player.awaiting_playback_completion = True 
                    self.standby_task = asyncio.create_task(self.standby_after_playback())
                    continue
                if event.type == "response.function_call_arguments.done":
                    print(f"Function call received: {event.name}")
                    sse_log_event("log", f"Function call received: {event.name} with args: {event.arguments}", log_type="system")
                    sse_log_event("active_tool_update", data={"name":event.name, "description":f"Executing tool with args: {event.arguments}"})
                    sse_log_event("status_update", data={"state":"Processing", "details":f"Executing tool: {event.name}", "last_event_message":f"Tool: {event.name}"})
                    await self.handle_function_call(event)
                    continue
                if event.type == "input_audio_buffer.speech_started":
                    print("User is speaking")
                    sse_log_event("log", "User started speaking.", log_type="user")
                    sse_log_event("status_update", data={"state":"Listening", "details":"User is speaking.", "last_event_message":"User speech started"})
                    if self.standby_task is not None:
                        self.standby_task.cancel()
                    self.standby_task = None
                    
                    if self.speech_end_delay_task and not self.speech_end_delay_task.done():
                        self.speech_end_delay_task.cancel()
                        self.speech_end_delay_task = None
                        print("Speech resumed - cancelling speech end delay")
                    if not self.should_send_audio.is_set():
                        self.should_send_audio.set()
                        self.is_recording = True 
                        sse_log_event("log", "User interrupted AI. Mic unmuted.", log_type="system")
                    continue
                
                if event.type == "input_audio_buffer.speech_stopped":
                    print("User paused speaking - waiting to confirm end of speech...")
                    sse_log_event("log", "User paused speaking.", log_type="user")
                    if user_transcript.strip():
                        self.last_user_transcript_for_tool_creation = user_transcript
                        sse_log_event("log", f"User (full transcript on pause): {user_transcript}", log_type="user")
                        print(f"User (full transcript on pause): {user_transcript}")
                        user_transcript = ""
                    
                    sse_log_event("status_update", data={"state":"Processing", "details":"User paused, waiting for speech end.", "last_event_message":"User speech paused"})
                    if not self.speech_end_delay_task or self.speech_end_delay_task.done():
                        self.speech_end_delay_task = asyncio.create_task(
                            self.delay_speech_end()
                        )
                    continue

    async def delay_speech_end(self):
        """Delay ending the speech recording to handle stutters or brief pauses."""
        try:
            await asyncio.sleep(self.speech_end_delay_seconds)
            print(f"No speech detected for {self.speech_end_delay_seconds}s - ending recording")
            sse_log_event("log", f"No speech detected for {self.speech_end_delay_seconds}s - ending recording.", log_type="system")
            sse_log_event("status_update", data={"state":"Processing", "details":"User speech ended, processing.", "last_event_message":"User speech ended"})
            self.should_send_audio.clear()
            self.is_recording = False
        except asyncio.CancelledError:
            sse_log_event("log", "Speech end delay cancelled (speech resumed).", log_type="system")
            pass

    async def standby_after_playback(self):
        """Wait for playback to complete and enter standby mode if no user response."""
        try:
            await self.playback_complete_event.wait()
            self.playback_complete_event.clear()
            
            print("Local audio playback complete. Waiting for user response...")
            sse_log_event("log", f"Local audio playback complete. Waiting {INACTIVITY_TIMEOUT}s for user response...", log_type="system")
            sse_log_event("status_update", data={"state":"Processing", "details":f"Playback complete. Waiting {INACTIVITY_TIMEOUT}s for user.", "last_event_message":"Playback complete"})

            if self.conversation_active:
                if not self.should_send_audio.is_set(): 
                    self.should_send_audio.set()
                    self.is_recording = True 
                    print("\nListening (OpenAI stream)...")
                    sse_log_event("log", "OpenAI audio input unmuted after AI audio playback. Listening for user.", log_type="system")
                    sse_log_event("status_update", data={"state":"Listening", "details":"Ready for user input.", "last_event_message":"Listening for user"})
                else:
                    print("\nAlready listening (OpenAI stream)...")
                    sse_log_event("status_update", data={"state":"Listening", "details":"Ready for user input.", "last_event_message":"Listening for user"})


            await asyncio.sleep(INACTIVITY_TIMEOUT)
            if self.conversation_active:
                self.enter_standby_mode() 
        except asyncio.CancelledError:
            print("Standby task canceled due to user activity.")
            sse_log_event("log", "Standby task after playback canceled due to user activity.", log_type="system")


    def enter_standby_mode(self):
        """Enter standby mode, ending the conversation and reactivating wake word detection."""
        self.conversation_active = False
        self.should_send_audio.clear()
        self.is_recording = False
        self.wake_word_detector.reset_cache()
        self.wake_word_active = True
        toggle_system_mic_mute_if_needed(False)
        print("Standby Mode activated. Listening for wake word...")
        sse_log_event("status_update", data={"state":"Standby", "details":"Listening for wake word.", "last_event_message":"Entered Standby Mode"})
        sse_log_event("active_tool_update", data={"name":"", "description":""})

    async def _get_connection(self) -> AsyncRealtimeConnection:
        """Get the current connection, waiting if necessary."""
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def process_wake_word_audio(self):
        """Process audio for wake word detection."""
        read_size = int(WAKE_WORD_SAMPLE_RATE * self.wake_word_detector.frame_duration)
        stream = sd.InputStream(channels=1, samplerate=WAKE_WORD_SAMPLE_RATE, dtype="int16")
        stream.start()
        try:
            while self.running:
                if self.wake_word_active:
                    if stream.read_available < read_size:
                        await asyncio.sleep(0.01)
                        continue
                    data, _ = stream.read(read_size)
                    frame = data.flatten()
                    self.wake_word_detector.process_audio_frame(frame)
                else:
                    await asyncio.sleep(0.01)
        finally:
            stream.stop()
            stream.close()

    async def send_mic_audio(self) -> None:
        """Record and send microphone audio to the API."""
        read_size = int(SAMPLE_RATE * 0.02)
        sent_audio = False
        stream = sd.InputStream(channels=CHANNELS, samplerate=SAMPLE_RATE, dtype="int16")
        stream.start()
        sse_log_event("log", "Microphone stream started for OpenAI.", log_type="system")
        try:
            while self.running:
                if stream.read_available < read_size:
                    await asyncio.sleep(0) 
                    continue
                if self.should_send_audio.is_set():
                    data, _ = stream.read(read_size)
                    connection = await self._get_connection()
                    if not sent_audio: 
                        asyncio.create_task(connection.send({"type": "response.cancel"})) 
                        sent_audio = True
                        sse_log_event("log", "Sending audio to OpenAI.", log_type="system")
                    await connection.input_audio_buffer.append(audio=base64.b64encode(cast(Any, data)).decode("utf-8"))
                else: 
                    if sent_audio: 
                        sent_audio = False
                        print("\nProcessing...")
                        sse_log_event("log", "Stopped sending audio. Processing...", log_type="system")
                        connection = await self._get_connection()
                        self.is_recording = False 
                await asyncio.sleep(0) 
        finally:
            stream.stop()
            stream.close()
            sse_log_event("log", "Microphone stream for OpenAI stopped.", log_type="system")

    async def handle_function_call(self, event) -> None:
        """Handle function calls from the AI."""
        try:
            name = event.name
            call_id = event.call_id
            arguments = json.loads(event.arguments)
            print(f"Function call: {name} with args: {arguments}")
            
            original_instruction = None
            if name == "learn_tool":
                original_instruction = arguments.get("instruction", "")
            
            result = None
            if name in self.tool_implementations:
                result = await self.tool_implementations[name](self, arguments)
            else:
                print(f"No implementation found for tool: {name}")
                sse_log_event("log", f"No implementation found for tool: {name}", log_type="error")
                result = f"Function {name} is not implemented."
                
            if name == "learn_tool" and result:
                try:
                    tool_match = re.search(r"Function '(\w+)' saved to", result)
                    if tool_match:
                        new_tool_name = tool_match.group(1)
                        print(f"Reloading tools after creating {new_tool_name}...")
                        sse_log_event("log", f"Reloading tools after creating {new_tool_name}...", log_type="system")
                        self.tools = self.load_tools_from_folder(self.tools_folder)
                        
                        connection = await self._get_connection()
                        await connection.session.update(tools=self.tools)
                        sse_log_event("log", "OpenAI session updated with new tools.", log_type="system")
                        
                        if self.last_user_transcript_for_tool_creation:
                            enhanced_result = (
                                f"Tool '{new_tool_name}' was successfully created. "
                                f"Now please use this new tool to fulfill the user's original request: "
                                f"'{self.last_user_transcript_for_tool_creation}'"
                            )
                            result = enhanced_result
                            self.last_user_transcript_for_tool_creation = ""
                except Exception as e:
                    print(f"Error processing learn_tool result: {e}")
            
            if result is not None:
                await self.send_function_call_result(result, call_id)
                sse_log_event("active_tool_update", data={"name":"", "description":""})
                sse_log_event("status_update", data={"state":"Responding", "details":"Assistant is formulating response.", "last_event_message":f"Tool {name} finished"})
        except Exception as e:
            print(f"Error handling function call: {e}")
            sse_log_event("log", f"Error handling function call: {e}", log_type="error")
            sse_log_event("active_tool_update", data={"name":"", "description":""})
            try:
                if 'call_id' in locals():
                    await self.send_function_call_result(f"Error: {str(e)}", call_id)
            except:
                pass

    async def send_function_call_result(self, result: str, call_id: str) -> None:
        """Send function call result back to OpenAI."""
        try:
            connection = await self._get_connection()
            await connection.send({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "output": result, "call_id": call_id}
            })
            await connection.send({"type": "response.create"})
            print(f"Sent function call result: {result}")
        except Exception as e:
            print(f"Failed to send function call result: {e}")

    async def run(self):
        signal.signal(signal.SIGINT, lambda sig, frame: self.handle_interrupt())
        print("Starting voice assistant. Press Ctrl+C to quit.")
        print("Say one of the wake words to activate: 'Hey Pluto' or 'Pluto'")
        sse_log_event("log", "Voice assistant run method started.", log_type="system")
        
        sse_server_thread = threading.Thread(target=start_sse_server, args=(8000,), daemon=True)
        sse_server_thread.start()

        connection_task = asyncio.create_task(self.handle_realtime_connection())
        mic_task = asyncio.create_task(self.send_mic_audio())
        wake_word_task = asyncio.create_task(self.process_wake_word_audio())
        await asyncio.gather(connection_task, mic_task, wake_word_task)
    
    def handle_interrupt(self):
        """Handle keyboard interrupt (Ctrl+C)."""
        print("\nShutting down gracefully...")
        toggle_system_mic_mute_if_needed(False)
        sse_log_event("log", "Shutdown initiated by user (Ctrl+C). System mic unmuted (toggled).", log_type="system")
        self.running = False
        self.should_send_audio.clear()
        if hasattr(self, 'audio_player') and self.audio_player:
            self.audio_player.stop()
            self.audio_player.terminate()
        if self.loop and not self.loop.is_closed():
            for task in asyncio.all_tasks(loop=self.loop):
                if task is not asyncio.current_task(loop=self.loop):
                    task.cancel()
        sse_log_event("log", "All asyncio tasks cancelled for shutdown.", log_type="system")

async def main():
    assistant = RealtimeVoiceAssistant()
    await assistant.run()

if __name__ == "__main__":
    sse_log_event("log", "Application starting...", log_type="system")
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sse_log_event("log", "OPENAI_API_KEY environment variable not set.", log_type="error")
        sys.exit(1)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Main loop interrupted by user.")
        sse_log_event("log", "Main loop interrupted by user.", log_type="system")
    finally:
        toggle_system_mic_mute_if_needed(False) 
        sse_log_event("log", "Application shutting down. System mic unmuted (toggled).", log_type="system")
        print("Application has shut down.")
