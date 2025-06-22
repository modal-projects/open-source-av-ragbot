#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Speech-to-speech model

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import sys

from loguru import logger
import aiohttp

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.turn.smart_turn.local_smart_turn import LocalSmartTurnAnalyzer
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport, SmallWebRTCConnection
from server.bot.animation import TalkingAnimation, get_frames

from ..services.stt.parakeet_service import ParakeetSTTService
from ..services.tts.chatterbox_service import ChatterboxTTSService
from ..services.rag.structured_rag_llm_service import StructuredRAGLLMService

try:
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
except ValueError:
    # Handle the case where logger is already initialized
    pass


# REPLACE WITH YOUR MODAL URL ENDPOINT
vllm_url = "https://modal-labs-shababo-dev--modal-rag-openai-vllm-vllmragser-cfa200.modal.run"
vllm_api_key = "super-secret-key" #os.getenv("VLLM_API_KEY", "super-secret-key")

chatterbox_url = "https://modal-labs-shababo-dev--chatterbox-tts-chatterbox-tts.modal.run"

async def run_bot(webrtc_connection: SmallWebRTCConnection):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport with specific audio parameters
    - Gemini Live multimodal model integration
    - Voice activity detection
    - Animation processing
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
            video_out_enabled=True,
            video_out_width=1024,
            video_out_height=576,
            video_out_framerate=12,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=0.2)
            ),
            turn_analyzer=LocalSmartTurnAnalyzer(
                smart_turn_model_path=None, # required kwarg, default model from HF is None (should be Optional not required!)
                params=SmartTurnParams(
                    stop_secs=2.0,
                    pre_speech_ms=0.0,
                    max_duration_secs=8.0
                )
            ),
        )

        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection, params=transport_params
        )

        markdown_filter = MarkdownTextFilter(
            MarkdownTextFilter.InputParams(
                filter_code=True,
                filter_tables=True
            )
        )
        
        stt = ParakeetSTTService(
            sample_rate=16000,
        )

        # Initialize LLM service
        params = OpenAILLMService.InputParams(
            extra = {
                "stream": True,
            }
        )
        llm = StructuredRAGLLMService(
            # To use OpenAI
            api_key=vllm_api_key,
            # Or, to use a local vLLM (or similar) api server
            model="modal-rag", #neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
            base_url=f"{vllm_url}/v1",
            params=params,
        )

        messages = [
            {
                "role": "user",
                "content": "Hi, could you please introduce yourself, including your name, and concisely tell me what you can do?" ,
            }
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)


        ta = TalkingAnimation()

        # Create structured output parser to extract answer_for_tts from JSON
        # structured_parser = StructuredOutputParser()
        
        # You can access extracted data at any time with:
        # data = structured_parser.get_extracted_data()
        # print(f"Code blocks: {data['code_blocks']}")
        # print(f"Links: {data['links']}")

        tts = ChatterboxTTSService(
            aiohttp_session=session,
            base_url=chatterbox_url,
            sample_rate=24000,
            text_filter=markdown_filter,
        )


        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                stt,
                context_aggregator.user(),
                llm,
                tts,
                ta,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                # enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )
        await task.queue_frame(get_frames("thinking"))

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            logger.info("Pipecat client ready.")
            await rtvi.set_bot_ready()
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Pipecat Client disconnected")
            await task.cancel()
            logger.info("Pipeline task cancelled.")

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Pipecat Client connected")

        @transport.event_handler("on_client_closed")
        async def on_client_closed(transport, client):
            logger.info("Pipecat Client closed")
            await task.cancel()
            logger.info("Pipeline task cancelled.")

        runner = PipelineRunner()

        await runner.run(task)
        logger.info("Pipeline task Finished.")