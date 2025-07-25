from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "chatterbox-streaming@git+https://github.com/shababo/chatterbox-streaming.git@80b514ef7bb6c8fb38856e9f635a3b51dafd6be9",
        "fastapi[standard]",
        "torchaudio",
        "transformers",
        "torch",
    )
    .add_local_dir(Path(__file__).parent / "assets", "/voice_samples")
)
app = modal.App("chatterbox-tts", image=image)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from fastapi.responses import StreamingResponse

@app.cls(
    gpu="h100", scaledown_window=60 * 5, enable_memory_snapshot=True, min_containers=1
)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter(snap=True)
    def load(self):
        self.model = ChatterboxTTS.from_pretrained(device="cuda")
        self.audio_prompt_path = "/voice_samples/kitt_voice_sample_converted_short_24000.wav"

        print("🔥 Warming up the model...")
        # warm up the model
        self.model.generate_stream(
            "Hello, how are you? We are Moe and Dal. We can help you learn more about Modal and developing with Modal. What can we help you with today?",
            audio_prompt_path=self.audio_prompt_path,
            chunk_size=25
        )

        print("✅ Model warmed up!")
    
    def _create_wav_header(self, sample_rate: int, channels: int, bits_per_sample: int, estimated_data_size: int) -> bytes:
        """Create a WAV file header for streaming. Much more efficient than using torchaudio.save for each chunk."""
        import struct
        
        # WAV header structure (44 bytes total)
        header = bytearray()
        
        # RIFF header
        header.extend(b'RIFF')
        header.extend(struct.pack('<I', 36 + estimated_data_size))  # File size - 8
        header.extend(b'WAVE')
        
        # fmt subchunk
        header.extend(b'fmt ')
        header.extend(struct.pack('<I', 16))  # Subchunk1Size (16 for PCM)
        header.extend(struct.pack('<H', 1))   # AudioFormat (1 for PCM)
        header.extend(struct.pack('<H', channels))  # NumChannels
        header.extend(struct.pack('<I', sample_rate))  # SampleRate
        header.extend(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))  # ByteRate
        header.extend(struct.pack('<H', channels * bits_per_sample // 8))  # BlockAlign
        header.extend(struct.pack('<H', bits_per_sample))  # BitsPerSample
        
        # data subchunk header
        header.extend(b'data')
        header.extend(struct.pack('<I', estimated_data_size))  # Subchunk2Size
        
        return bytes(header)

    @modal.fastapi_endpoint(docs=True, method="POST")
    def tts(self, prompt: str):
        import time
        
        # Generate streaming audio from the input text
        print(f"🎤 Starting streaming generation for prompt: {prompt}")
        
        stream_start = time.time()
        try:
            stream_generator = self.model.generate_stream(
                prompt, 
                audio_prompt_path=self.audio_prompt_path,
                chunk_size=25  # Smaller chunks for lower latency
            )
            
            # Wrap the generator to add timing and efficient tensor-to-bytes conversion
            def timed_stream_generator():
                chunk_count = 0
                first_chunk_time = None
                header_sent = False
                
                for chunk, metric in stream_generator:
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        print(f"⏱️  Time to first chunk: {first_chunk_time - stream_start:.3f} seconds")
                    
                    chunk_count += 1
                    if chunk_count % 10 == 0:  # Log every 10th chunk
                        print(f"📊 Streamed {chunk_count} chunks so far")
                    
                    # Convert torch tensor to bytes efficiently
                    try:
                        # Handle tensor format - might be (batch, samples) or just (samples,)
                        if chunk.dim() > 1:
                            audio_tensor = chunk[0]  # Take first batch if batched
                        else:
                            audio_tensor = chunk
                        
                        # Ensure tensor is on CPU and convert to numpy for efficiency
                        audio_numpy = audio_tensor.cpu().numpy()
                        
                        # Convert float32 audio to int16 PCM (standard for WAV)
                        # Clamp to [-1, 1] range and scale to int16 range
                        audio_numpy = audio_numpy.clip(-1.0, 1.0)
                        pcm_data = (audio_numpy * 32767).astype('int16')
                        
                        if not header_sent:
                            # Send WAV header only for the first chunk
                            # This is much more efficient than full WAV file per chunk
                            wav_header = self._create_wav_header(
                                sample_rate=self.model.sr,
                                channels=1,
                                bits_per_sample=16,
                                # Use a large data size for streaming (will be ignored by most players)
                                estimated_data_size=1000000  
                            )
                            yield wav_header + pcm_data.tobytes()
                            header_sent = True
                        else:
                            # For subsequent chunks, just send raw PCM data
                            yield pcm_data.tobytes()
                        
                    except Exception as e:
                        print(f"❌ Error converting chunk {chunk_count}: {e}")
                        print(f"   Chunk shape: {chunk.shape if hasattr(chunk, 'shape') else 'N/A'}")
                        print(f"   Chunk type: {type(chunk)}")
                        continue  # Skip this chunk and continue
                
                final_time = time.time()
                print(f"⏱️  Total streaming time: {final_time - stream_start:.3f} seconds")
                print(f"📊 Total chunks streamed: {chunk_count}")
                print("✅ Chatterbox streaming complete!")
            
            wrapped_generator = timed_stream_generator()
            
        except Exception as e:
            print(f"❌ Error creating stream generator: {e}")
            raise


        # Return the audio as a streaming response with appropriate MIME type.
        # This allows for browsers to playback audio directly.
        return StreamingResponse(
            content=wrapped_generator,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="output.wav"'},
        )

def get_chatterbox_server_url():
    try:
        return Chatterbox().tts.get_web_url()
    except Exception as e:
        try:
            ChatterboxCls = modal.Cls.from_name("chatterbox-tts", "Chatterbox")
            return ChatterboxCls().tts.get_web_url()
        except Exception as e:
            print(f"❌ Error getting Chatterbox server URL: {e}")
            return None