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

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams
from pipecat.frames.frames import LLMRunFrame

from .services.parakeet_service import ModalSegmentedSTTService
from .services.kokoro_websocket_service import KokoroTTSService
from .services.modal_rag_service import ModalRagLLMService
from .processors.text_aggregator import ModalRagTextAggregator
from .processors.modal_rag import ModalRag, get_system_prompt

from .avatar.animation import MoeDalBotAnimation
from .processors.parser import ModalRagStreamingJsonParser

import modal

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass

_AUDIO_INPUT_SAMPLE_RATE = 16000
_AUDIO_OUTPUT_SAMPLE_RATE = 24000
_MOE_AND_DAL_FRAME_RATE = 12
_MOE_AND_DAL_FRAME_WIDTH = 1024
_MOE_AND_DAL_FRAME_HEIGHT = 576


async def run_bot(
    webrtc_connection: SmallWebRTCConnection,
    enable_video: bool = False,
):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - WebRTC transport with audio/video parameters
    - Structured RAG LLM service with vLLM backend
    - Parakeet STT and Chatterbox TTS services
    - Voice activity detection and smart turn management
    - Animation processing with talking animations
    - RTVI event handling
    """

    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=_AUDIO_INPUT_SAMPLE_RATE,
        audio_out_enabled=True,
        audio_out_sample_rate=_AUDIO_OUTPUT_SAMPLE_RATE,
        video_out_enabled=enable_video,
        video_out_width=_MOE_AND_DAL_FRAME_WIDTH,
        video_out_height=_MOE_AND_DAL_FRAME_HEIGHT,
        video_out_framerate=_MOE_AND_DAL_FRAME_RATE,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.2)
        ),
        turn_analyzer=LocalSmartTurnAnalyzerV3(
            params=SmartTurnParams()
        ),
    )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=transport_params,
    )

    stt = ModalSegmentedSTTService(
        app_name="parakeet-transcription",
        cls_name="Transcriber",
        audio_passthrough=True,
    )

    modal_rag = ModalRag(similarity_top_k=5)

    func = modal.Function.from_name("vllm-service", "serve")
    llm_url = func.get_web_url() + "/v1"

    # Initialize OpenAI API compatibleLLM service
    # llm = OpenAILLMService(
    #     model="Qwen/Qwen3-4B-Instruct-2507",
    #     api_key = "super-secret-key",
    #     base_url = llm_url,
    #     params=OpenAILLMService.InputParams(
    #         extra={
    #             "stream": True,
    #         },
    #     ),
    # )

    # json_parser = ModalRagStreamingJsonParser()

    llm = ModalRagLLMService(
        model="Qwen/Qwen3-4B-Instruct-2507",
        api_key = "super-secret-key",
        base_url = llm_url,
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

    if enable_video:
        ta = MoeDalBotAnimation()
    else:
        ta = None

    tts = KokoroTTSService(
        text_aggregator=ModalRagTextAggregator(),
        # pause_frame_processing=True,
    )

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))


    processors = [
        transport.input(),
        stt,
        modal_rag,
        context_aggregator.user(),
        llm,
        # json_parser,
        tts,
        rtvi,
    ]
    if enable_video: # only add animation processor if video is enabled
        processors.append(ta)
    processors += [
        transport.output(),
        context_aggregator.assistant(),
    ]
    pipeline = Pipeline(
        processors=processors,
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
