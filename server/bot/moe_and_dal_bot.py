# Built from Pipecat's Modal Deployment example here:
# https://github.com/pipecat-ai/pipecat/tree/main/examples/deployment/modal-example


"""Moe and Dal RAG Bot Implementation.

This module implements a chatbot using a custom RAG (Retrieval-Augmented Generation)
system with VLLM backend. It includes:
- Real-time audio/video interaction through WebRTC
- Animated robot avatar with talking animations
- Speech-to-speech pipeline with Parakeet STT and Chatterbox TTS
- Structured RAG LLM service for enhanced responses
- Voice activity detection and smart turn management

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using the RAG system's streaming capabilities.
"""

import sys

from loguru import logger
import aiohttp
import datetime
import io
import os
import wave
import aiofiles

from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.transports.base_transport import TransportParams
# from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
# from pipecat.audio.filters.noisereduce_filter import NoisereduceFilter
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.frames.frames import LLMRunFrame

from server.bot.animation import MoeDalBotAnimation, get_frames
# from server.bot.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
# from ..services.stt.kyutai_service import KyutaiSTTService
from ..services.stt.parakeet_service import ParakeetSTTService

from ..services.modal_rag.modal_rag_service import ModalRagLLMService, get_system_prompt


# from ..services.tts.chatterbox_service import ChatterboxTTSService
# from ..services.tts.kokoro_service import KokoroTTSService
from ..services.tts.kokoro_websocket_service import KokoroTTSService
from ..services.tts.text_aggregator import ModalRagTextAggregator
from .modal_rag import ModalRag

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass

_AUDIO_INPUT_SAMPLE_RATE = 16000
_AUDIO_OUTPUT_SAMPLE_RATE = 24000
_MOE_AND_DAL_FRAME_RATE = 12

async def save_audio_file(audio: bytes, filename: str, sample_rate: int, num_channels: int):
    """Save audio data to a WAV file."""
    if len(audio) > 0:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Audio saved to {filename}")

async def run_bot(webrtc_connection: SmallWebRTCConnection):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - WebRTC transport with audio/video parameters
    - Structured RAG LLM service with vLLM backend
    - Parakeet STT and Chatterbox TTS services
    - Voice activity detection and smart turn management
    - Animation processing with talking animations
    - RTVI event handling
    """
    # async with aiohttp.ClientSession() as session:
    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=_AUDIO_INPUT_SAMPLE_RATE,
        audio_out_enabled=True,
        audio_out_sample_rate=_AUDIO_OUTPUT_SAMPLE_RATE,
        # audio_out_10ms_chunks=8,
        # audio_in_filter=NoisereduceFilter(),
        video_out_enabled=False,
        video_out_width=1024,
        video_out_height=576,
        video_out_framerate=_MOE_AND_DAL_FRAME_RATE,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.2)
        ),
        turn_analyzer=LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams(
                stop_secs=0.250,
                pre_speech_ms=0.0,
                max_duration_secs=8.0
            )
        ),
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=transport_params,
    )


    stt = ParakeetSTTService(
        # sample_rate=24000,
        audio_passthrough=True,
    )

    modal_rag = ModalRag(similarity_top_k=5)

    # Initialize OpenAI API compatibleLLM service
    llm = ModalRagLLMService(
        # model="modal-rag",
        model="Qwen/Qwen3-4B-Instruct-2507",
        params=OpenAILLMService.InputParams(
            extra={
                "stream": True,
            },
        ),
    )

    messages = [
        {
            "role": "system", 
            "content": get_system_prompt(),
        },
        {
            "role": "user",
            "content": "Hi, could the two of you introduce yourselves?",
        }
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    # ta = MoeDalBotAnimation()

    tts = KokoroTTSService(
        # aiohttp_session=session,
        # sample_rate=24000,
        text_aggregator=ModalRagTextAggregator(),
        # pause_frame_processing=True,
    )

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    
    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            modal_rag,
            context_aggregator.user(),
            llm,
            tts,
            # ta,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            report_only_initial_ttfb=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        # await task.queue_frames([context_aggregator.user().get_context_frame()])
        await task.queue_frame(get_frames("thinking"))
        await task.queue_frame(LLMRunFrame())
        
        
        

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()
        logger.info("Pipeline task cancelled.")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")
        # Kick off the conversation
        # await task.queue_frame(LLMRunFrame())
        

    

    runner = PipelineRunner()
    await runner.run(task)

    logger.info("Pipeline task Finished.")
