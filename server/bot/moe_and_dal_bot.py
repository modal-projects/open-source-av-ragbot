import sys
import asyncio
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.parallel_pipeline import ParallelPipeline

from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor

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

from .services.modal_services import ModalTunnelManager
from .services.modal_parakeet_service import ModalParakeetSegmentedSTTService
from .services.modal_kokoro_service import ModalKokoroTTSService
from .processors.unison_speaker_mixer import UnisonSpeakerMixer
from .services.modal_openai_service import ModalOpenAILLMService
from .processors.modal_rag import ModalRag, get_system_prompt, ChromaVectorDB

from .avatar.animation import MoeDalBotAnimation, get_frames

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

_DEFAULT_ENABLE_VIDEO = False

async def run_bot(
    webrtc_connection: SmallWebRTCConnection,
    chroma_db: ChromaVectorDB,
    enable_moe_and_dal: bool = _DEFAULT_ENABLE_VIDEO,
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

    # spawn services first (happens on modaltunnelmanager init)
    sglang_tunnel_manager = ModalTunnelManager(
        app_name="sglang-server",
        cls_name="SGLangServer",
    )

    # get_llm_service_task = asyncio.create_task(
    #     ModalOpenAILLMService.from_tunnel_manager(
    #         model="Qwen/Qwen3-4B-Instruct-2507",
    #         modal_tunnel_manager=sglang_tunnel_manager,
    #         params=OpenAILLMService.InputParams(
    #             extra={
    #                 "stream": True,
    #             },
    #         ),
    #     )
    # )

    llm = ModalOpenAILLMService(
        model="Qwen/Qwen3-4B-Instruct-2507",
        modal_tunnel_manager=sglang_tunnel_manager,
        params=OpenAILLMService.InputParams(
            extra={
                "stream": True,
            },
        ),
    )

    parakeet_stt_tunnel_manager=ModalTunnelManager(
        app_name="parakeet-transcription",
        cls_name="Transcriber",
    )
    if enable_moe_and_dal:
        kokoro_tts_tunnel_manager_moe = ModalTunnelManager(
            app_name="kokoro-tts",
            cls_name="KokoroTTS",
        )
        kokoro_tts_tunnel_manager_dal = ModalTunnelManager(
            app_name="kokoro-tts",
            cls_name="KokoroTTS",
        )
    else:
        kokoro_tts_tunnel_manager = ModalTunnelManager(
            app_name="kokoro-tts",
            cls_name="KokoroTTS",
        )

    transport_params = TransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=_AUDIO_INPUT_SAMPLE_RATE,
        audio_out_enabled=True,
        audio_out_sample_rate=_AUDIO_OUTPUT_SAMPLE_RATE,
        video_out_enabled=enable_moe_and_dal,
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

    stt = ModalParakeetSegmentedSTTService(
        modal_tunnel_manager=parakeet_stt_tunnel_manager,
    )

    modal_rag = ModalRag(chroma_db=chroma_db, similarity_top_k=3, num_adjacent_nodes=2)

     # only add animation processor and dual speaker setup if video is enabled
    if enable_moe_and_dal:
        ta = MoeDalBotAnimation()
        moe_tts = ModalKokoroTTSService(
            modal_tunnel_manager=kokoro_tts_tunnel_manager_moe,
            speaker="moe",
            voice="am_puck",
            speed=1.3,
        )
        dal_tts = ModalKokoroTTSService(
            modal_tunnel_manager=kokoro_tts_tunnel_manager_dal,
            speaker="dal",
            voice="am_fenrir",
            speed=1.5,
        )
        speaker_mixer = UnisonSpeakerMixer(speakers=["moe", "dal"])
    else:
        tts = ModalKokoroTTSService(
            modal_tunnel_manager=kokoro_tts_tunnel_manager,
            voice="am_puck",
            speed=1.35,
        )


    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor()

    messages = [
        {
            "role": "system", 
            "content": get_system_prompt(enable_moe_and_dal=enable_moe_and_dal),
        },
        {
            "role": "user",
            "content": "Hi, could the two of you introduce yourselves?",
        }
    ]

    # Set up conversation context and management
    # The context_aggregator will automatically collect conversation context
    context = OpenAILLMContext(messages)

    # llm = await get_llm_service_task
    context_aggregator = llm.create_context_aggregator(
        context,
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.05),
    )

    processors = [
        transport.input(),
        rtvi,
        stt,
        modal_rag,
        context_aggregator.user(),
        llm,
        
    ]
    if enable_moe_and_dal:
        processors += [
            ParallelPipeline(
                [moe_tts],
                [dal_tts],
            ),
            speaker_mixer,
            ta,
        ]
    else:
        processors.append(tts)

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
        await task.queue_frame(get_frames("listening"))
        
    runner = PipelineRunner()
    await runner.run(task)

    logger.info("Pipeline task Finished.")
