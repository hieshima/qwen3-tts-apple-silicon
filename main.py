import os
import sys
import shutil
import time
import wave
import gc
import re
import subprocess
import warnings
from datetime import datetime

# Suppress harmless library warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import types
    import mlx.core as mx
    from tqdm import tqdm
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.models.base import GenerationResult
except ImportError:
    print("Error: 'mlx_audio' library not found.")
    print("Run: source .venv/bin/activate")
    sys.exit(1)


def _format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _optimized_generate_with_instruct(
    self, text, speaker, language, instruct, temperature, max_tokens,
    top_k, top_p, repetition_penalty, verbose, stream=False, streaming_interval=2.0,
):
    """Optimized generation with single GPU-CPU sync per token step."""
    if self.speech_tokenizer is None:
        raise ValueError("Speech tokenizer not loaded")

    start_time = time.time()

    input_embeds, trailing_text_hidden, tts_pad_embed = (
        self._prepare_generation_inputs(
            text=text, language=language, speaker=speaker, instruct=instruct,
        )
    )

    target_token_count = len(self.tokenizer.encode(text))
    effective_max_tokens = min(max_tokens, max(75, target_token_count * 6))

    cache = self.talker.make_cache()
    generated_codes = []
    generated_token_history = []  # Incremental tracking instead of O(n^2) rebuild
    config = self.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = [
        i for i in range(config.vocab_size - 1024, config.vocab_size)
        if i != eos_token_id
    ]
    trailing_idx = 0

    streaming_chunk_size = max(1, int(streaming_interval * 12.5))
    decoded_tokens = 0
    context_size = 25

    pbar = tqdm(
        total=effective_max_tokens, desc="Generating", unit="tokens",
        disable=not verbose, leave=False,
    )

    profiling = os.environ.get("TTS_PROFILE") == "1"
    profile_steps = 10  # Only print first N steps

    for step in range(effective_max_tokens):
        if profiling:
            step_t0 = time.perf_counter()

        # Talker forward pass
        logits, hidden = self.talker(input_embeds, cache=cache)

        if profiling:
            mx.eval(logits, hidden)
            t_talker = time.perf_counter()

        # Sample first codebook token
        next_token = self._sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated_token_history or None,
            suppress_tokens=suppress_tokens, eos_token_id=eos_token_id,
        )

        if profiling:
            mx.eval(next_token)
            t_sample = time.perf_counter()

        # Code predictor for remaining 15 codebooks
        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        code_cache = self.talker.code_predictor.make_cache()

        for code_idx in range(config.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = self.talker.get_input_embeddings()(next_token)
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_embed = self.talker.code_predictor.codec_embedding[
                    code_idx - 1
                ](code_tokens[-1])
                code_input = code_embed

            code_logits, code_cache, _ = self.talker.code_predictor(
                code_input, cache=code_cache, generation_step=code_idx,
            )

            next_code = self._sample_token(
                code_logits, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            code_tokens.append(next_code)

        all_codes = mx.concatenate(code_tokens, axis=1)

        if profiling:
            mx.eval(all_codes)
            t_codepred = time.perf_counter()

        # Prepare next input embedding
        if trailing_idx < trailing_text_hidden.shape[1]:
            text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed

        codec_embed = self.talker.get_input_embeddings()(next_token)
        for i, code in enumerate(code_tokens[1:]):
            codec_embed = (
                codec_embed + self.talker.code_predictor.codec_embedding[i](code)
            )
        input_embeds = text_embed + codec_embed

        if profiling:
            mx.eval(input_embeds)
            t_embed = time.perf_counter()
            if step < profile_steps or step % 100 == 0:
                print(f"  Step {step}: talker={1000*(t_talker-step_t0):.1f}ms"
                      f"  sample0={1000*(t_sample-t_talker):.1f}ms"
                      f"  codepred={1000*(t_codepred-t_sample):.1f}ms"
                      f"  embed={1000*(t_embed-t_codepred):.1f}ms"
                      f"  total={1000*(t_embed-step_t0):.1f}ms")
        else:
            # === SINGLE GPU-CPU SYNC for the entire step ===
            mx.eval(input_embeds, next_token, all_codes)

        # EOS check (zero overhead - already evaluated)
        token_val = int(next_token[0, 0])
        if token_val == eos_token_id:
            break

        generated_token_history.append(token_val)
        generated_codes.append(all_codes)
        del code_cache

        # Periodic memory cleanup only
        if step > 0 and step % 100 == 0:
            mx.clear_cache()

        pbar.update(1)

        # Streaming support
        new_tokens = len(generated_codes) - decoded_tokens
        if stream and new_tokens >= streaming_chunk_size:
            start_idx = max(0, decoded_tokens - context_size)
            codes_chunk = mx.stack(generated_codes[start_idx:], axis=1)
            mx.eval(codes_chunk)

            audio_chunk = self._decode_chunk(codes_chunk, chunk_tokens=streaming_chunk_size)

            if decoded_tokens > 0 and start_idx < decoded_tokens:
                context_tokens_count = decoded_tokens - start_idx
                samples_per_token = self.speech_tokenizer.decode_upsample_rate
                trim_samples = context_tokens_count * samples_per_token
                if trim_samples < audio_chunk.shape[0]:
                    audio_chunk = audio_chunk[trim_samples:]

            decoded_tokens = len(generated_codes)

            yield GenerationResult(
                audio=audio_chunk, samples=audio_chunk.shape[0],
                sample_rate=self.sample_rate, segment_idx=0,
                token_count=new_tokens,
                audio_duration=_format_duration(audio_chunk.shape[0] / self.sample_rate),
                real_time_factor=0,
                prompt={"tokens": new_tokens, "tokens-per-sec": 0},
                audio_samples={"samples": audio_chunk.shape[0], "samples-per-sec": 0},
                processing_time_seconds=0,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
                is_streaming_chunk=True,
            )
            mx.clear_cache()

    pbar.close()

    # Streaming: yield remaining tokens
    if stream and len(generated_codes) > decoded_tokens:
        start_idx = max(0, decoded_tokens - context_size)
        codes_chunk = mx.stack(generated_codes[start_idx:], axis=1)
        mx.eval(codes_chunk)
        audio_chunk = self._decode_chunk(codes_chunk, chunk_tokens=streaming_chunk_size)
        if decoded_tokens > 0 and start_idx < decoded_tokens:
            context_tokens_count = decoded_tokens - start_idx
            samples_per_token = self.speech_tokenizer.decode_upsample_rate
            trim_samples = context_tokens_count * samples_per_token
            if trim_samples < audio_chunk.shape[0]:
                audio_chunk = audio_chunk[trim_samples:]
        new_tokens = len(generated_codes) - decoded_tokens
        yield GenerationResult(
            audio=audio_chunk, samples=audio_chunk.shape[0],
            sample_rate=self.sample_rate, segment_idx=0, token_count=new_tokens,
            audio_duration=_format_duration(audio_chunk.shape[0] / self.sample_rate),
            real_time_factor=0,
            prompt={"tokens": new_tokens, "tokens-per-sec": 0},
            audio_samples={"samples": audio_chunk.shape[0], "samples-per-sec": 0},
            processing_time_seconds=0,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=True, is_final_chunk=True,
        )
        return

    if not generated_codes:
        return

    # Non-streaming: decode all at once
    codes = mx.stack(generated_codes, axis=1)
    audio, audio_lengths = self.speech_tokenizer.decode(codes)
    audio = audio[0]
    valid_len = int(audio_lengths[0])
    if valid_len > 0 and valid_len < audio.shape[0]:
        audio = audio[:valid_len]
    mx.eval(audio)

    elapsed_time = time.time() - start_time
    samples = audio.shape[0]
    token_count = len(generated_codes)
    duration_seconds = samples / self.sample_rate
    rtf = duration_seconds / elapsed_time if elapsed_time > 0 else 0

    yield GenerationResult(
        audio=audio, samples=samples, sample_rate=self.sample_rate,
        segment_idx=0, token_count=token_count,
        audio_duration=_format_duration(duration_seconds),
        real_time_factor=rtf,
        prompt={
            "tokens": token_count,
            "tokens-per-sec": token_count / elapsed_time if elapsed_time > 0 else 0,
        },
        audio_samples={
            "samples": samples,
            "samples-per-sec": samples / elapsed_time if elapsed_time > 0 else 0,
        },
        processing_time_seconds=elapsed_time,
        peak_memory_usage=mx.get_peak_memory() / 1e9,
    )
    mx.clear_cache()


def _optimized_generate_icl(
    self, text, ref_audio, ref_text, language="auto", temperature=0.9,
    max_tokens=4096, top_k=50, top_p=1.0, repetition_penalty=1.5, verbose=False,
):
    """Optimized ICL generation with single GPU-CPU sync per token step."""
    if self.speech_tokenizer is None:
        raise ValueError("Speech tokenizer not loaded")

    start_time = time.time()

    input_embeds, trailing_text_hidden, tts_pad_embed, ref_codes = (
        self._prepare_icl_generation_inputs(
            text=text, ref_audio=ref_audio, ref_text=ref_text, language=language,
        )
    )

    target_token_count = len(self.tokenizer.encode(text))
    effective_max_tokens = min(max_tokens, max(75, target_token_count * 6))

    cache = self.talker.make_cache()
    generated_codes = []
    generated_token_history = []
    config = self.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = [
        i for i in range(config.vocab_size - 1024, config.vocab_size)
        if i != eos_token_id
    ]
    trailing_idx = 0

    pbar = tqdm(
        total=effective_max_tokens, desc="ICL Generation", unit="tokens",
        disable=not verbose, leave=False,
    )

    for step in range(effective_max_tokens):
        logits, hidden = self.talker(input_embeds, cache=cache)

        next_token = self._sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated_token_history or None,
            suppress_tokens=suppress_tokens, eos_token_id=eos_token_id,
        )

        code_tokens = [next_token]
        code_hidden = hidden[:, -1:, :]
        code_cache = self.talker.code_predictor.make_cache()

        for code_idx in range(config.num_code_groups - 1):
            if code_idx == 0:
                code_0_embed = self.talker.get_input_embeddings()(next_token)
                code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
            else:
                code_embed = self.talker.code_predictor.codec_embedding[
                    code_idx - 1
                ](code_tokens[-1])
                code_input = code_embed

            code_logits, code_cache, _ = self.talker.code_predictor(
                code_input, cache=code_cache, generation_step=code_idx,
            )

            next_code = self._sample_token(
                code_logits, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            code_tokens.append(next_code)

        all_codes = mx.concatenate(code_tokens, axis=1)

        if trailing_idx < trailing_text_hidden.shape[1]:
            text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
            trailing_idx += 1
        else:
            text_embed = tts_pad_embed

        codec_embed = self.talker.get_input_embeddings()(next_token)
        for i, code in enumerate(code_tokens[1:]):
            codec_embed = (
                codec_embed + self.talker.code_predictor.codec_embedding[i](code)
            )
        input_embeds = text_embed + codec_embed

        # Single GPU-CPU sync
        mx.eval(input_embeds, next_token, all_codes)

        token_val = int(next_token[0, 0])
        if token_val == eos_token_id:
            break

        generated_token_history.append(token_val)
        generated_codes.append(all_codes)
        del code_cache

        if step > 0 and step % 100 == 0:
            mx.clear_cache()

        pbar.update(1)

    pbar.close()

    if not generated_codes:
        return

    # ICL decode: prepend ref_codes to generated codes
    gen_codes = mx.stack(generated_codes, axis=1)
    ref_codes_t = mx.transpose(ref_codes, (0, 2, 1))
    full_codes = mx.concatenate([ref_codes_t, gen_codes], axis=1)

    ref_len = ref_codes.shape[2]
    total_len = full_codes.shape[1]

    audio, audio_lengths = self.speech_tokenizer.decode(full_codes)
    audio = audio[0]

    valid_len = int(audio_lengths[0])
    if valid_len > 0 and valid_len < audio.shape[0]:
        audio = audio[:valid_len]

    # Proportional trim to remove reference audio portion
    cut = int(ref_len / max(total_len, 1) * audio.shape[0])
    if cut > 0 and cut < audio.shape[0]:
        audio = audio[cut:]

    mx.eval(audio)

    elapsed_time = time.time() - start_time
    samples = audio.shape[0]
    token_count = len(generated_codes)
    duration_seconds = samples / self.sample_rate
    rtf = duration_seconds / elapsed_time if elapsed_time > 0 else 0

    yield GenerationResult(
        audio=audio, samples=samples, sample_rate=self.sample_rate,
        segment_idx=0, token_count=token_count,
        audio_duration=_format_duration(duration_seconds),
        real_time_factor=rtf,
        prompt={
            "tokens": token_count,
            "tokens-per-sec": token_count / elapsed_time if elapsed_time > 0 else 0,
        },
        audio_samples={
            "samples": samples,
            "samples-per-sec": samples / elapsed_time if elapsed_time > 0 else 0,
        },
        processing_time_seconds=elapsed_time,
        peak_memory_usage=mx.get_peak_memory() / 1e9,
    )
    mx.clear_cache()


def _ablation_revert_rmsnorm():
    """Revert RMSNorm to manual implementation (for A/B testing)."""
    from mlx_audio.tts.models.qwen3_tts.talker import RMSNorm

    def manual_rmsnorm(self, x):
        x_float = x.astype(mx.float32)
        variance = mx.mean(x_float**2, axis=-1, keepdims=True)
        x_normed = x_float * mx.rsqrt(variance + self.eps)
        return (self.weight * x_normed).astype(x.dtype)

    RMSNorm.__call__ = manual_rmsnorm
    print("[ABLATION] RMSNorm reverted to manual (unfused)")


def _ablation_revert_rope(model):
    """Revert code predictor RoPE to manual implementation (for A/B testing)."""
    from mlx_audio.tts.models.qwen3_tts.talker import (
        CodePredictorAttention, CodePredictorDecoderLayer, CodePredictorModel,
        apply_rotary_pos_emb,
    )

    def attn_call_manual_rope(self, x, position_embeddings, mask=None, cache=None):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        output = mx.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, -1)
        return self.o_proj(output)

    def layer_call_manual_rope(self, x, position_embeddings, mask=None, cache=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_embeddings, mask, cache)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

    import mlx.nn as nn_mlx
    def model_call_manual_rope(self, inputs_embeds, position_ids=None, mask=None, cache=None):
        batch, seq_len, _ = inputs_embeds.shape
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
        if position_ids is None:
            position_ids = mx.arange(offset, offset + seq_len)[None, :]
            position_ids = mx.broadcast_to(position_ids, (batch, seq_len))
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        if mask is None and seq_len > 1:
            mask = nn_mlx.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(inputs_embeds.dtype)
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, position_embeddings, mask, layer_cache)
        x = self.norm(x)
        return x

    CodePredictorAttention.__call__ = attn_call_manual_rope
    CodePredictorDecoderLayer.__call__ = layer_call_manual_rope
    CodePredictorModel.__call__ = model_call_manual_rope
    print("[ABLATION] Code predictor RoPE reverted to manual (unfused)")


def _apply_fused_rmsnorm():
    """Replace manual RMSNorm with mx.fast.rms_norm (single fused Metal kernel)."""
    from mlx_audio.tts.models.qwen3_tts.talker import RMSNorm
    RMSNorm.__call__ = lambda self, x: mx.fast.rms_norm(x, self.weight, self.eps)


def _apply_fused_rope():
    """Replace Code Predictor's manual RoPE with mx.fast.rope."""
    from mlx_audio.tts.models.qwen3_tts.talker import (
        CodePredictorAttention, CodePredictorDecoderLayer, CodePredictorModel,
    )
    import mlx.nn as nn_mlx

    def attn_call_fused_rope(self, x, offset=0, mask=None, cache=None):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.config.rope_theta, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        output = mx.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch, seq_len, -1)
        return self.o_proj(output)

    def layer_call_fused_rope(self, x, offset=0, mask=None, cache=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, offset, mask, cache)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

    def model_call_fused_rope(self, inputs_embeds, position_ids=None, mask=None, cache=None):
        batch, seq_len, _ = inputs_embeds.shape
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0].offset
        if mask is None and seq_len > 1:
            mask = nn_mlx.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(inputs_embeds.dtype)
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x = layer(x, offset, mask, layer_cache)
        x = self.norm(x)
        return x

    CodePredictorAttention.__call__ = attn_call_fused_rope
    CodePredictorDecoderLayer.__call__ = layer_call_fused_rope
    CodePredictorModel.__call__ = model_call_fused_rope


def optimize_model(model):
    """Apply all inference optimizations via monkey-patching (never modifies .venv)."""
    # 1. Fuse RMSNorm: mx.fast.rms_norm replaces manual 5-kernel implementation
    _apply_fused_rmsnorm()

    # 2. Fuse Code Predictor RoPE: mx.fast.rope replaces manual RotaryEmbedding
    _apply_fused_rope()

    # 3. Single GPU-CPU sync per token step
    if hasattr(model, '_generate_with_instruct'):
        model._generate_with_instruct = types.MethodType(
            _optimized_generate_with_instruct, model
        )
    if hasattr(model, '_generate_icl'):
        model._generate_icl = types.MethodType(
            _optimized_generate_icl, model
        )

    # A/B testing: ABLATION env var reverts specific optimizations
    ablation = os.environ.get("ABLATION", "").lower()
    if ablation == "rmsnorm":
        _ablation_revert_rope(model)
    elif ablation == "rope":
        _ablation_revert_rmsnorm()
    elif ablation == "baseline":
        _ablation_revert_rmsnorm()
        _ablation_revert_rope(model)
    elif ablation:
        print(f"[ABLATION] Unknown: '{ablation}'. Use: rmsnorm, rope, baseline.")

    return model

# Configuration
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
MODELS_DIR = os.path.join(os.getcwd(), "models")
VOICES_DIR = os.path.join(os.getcwd(), "voices")

# Settings
AUTO_PLAY = True
SAMPLE_RATE = 24000
FILENAME_MAX_LEN = 20

# Model Definitions
MODELS = {
    # Pro (1.7B, 8-bit)
    "1": {"name": "Custom Voice", "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "mode": "custom", "output_subfolder": "CustomVoice"},
    "2": {"name": "Voice Design", "folder": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit", "mode": "design", "output_subfolder": "VoiceDesign"},
    "3": {"name": "Voice Cloning", "folder": "Qwen3-TTS-12Hz-1.7B-Base-8bit", "mode": "clone_manager", "output_subfolder": "Clones"},
    # Lite (0.6B, 8-bit)
    "4": {"name": "Custom Voice", "folder": "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit", "mode": "custom", "output_subfolder": "CustomVoice"},
    "5": {"name": "Voice Design", "folder": "Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit", "mode": "design", "output_subfolder": "VoiceDesign"},
    "6": {"name": "Voice Cloning", "folder": "Qwen3-TTS-12Hz-0.6B-Base-8bit", "mode": "clone_manager", "output_subfolder": "Clones"},
    # Pro (1.7B, 4-bit) — faster, slightly lower quality
    "7": {"name": "Custom Voice (4-bit)", "folder": "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit", "mode": "custom", "output_subfolder": "CustomVoice"},
}

SPEAKER_MAP = {
    "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean": ["Sohee"]
}

EMOTION_EXAMPLES = [
    "Sad and crying, speaking slowly",
    "Excited and happy, speaking very fast",
    "Angry and shouting",
    "Whispering quietly"
]


def flush_input():
    try:
        import termios
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)
    except (ImportError, OSError):
        pass


def clean_memory():
    gc.collect()


def make_temp_dir():
    return f"temp_{int(time.time())}"


def get_smart_path(folder_name):
    full_path = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(full_path):
        return None

    snapshots_dir = os.path.join(full_path, "snapshots")
    if os.path.exists(snapshots_dir):
        subfolders = [f for f in os.listdir(snapshots_dir) if not f.startswith('.')]
        if subfolders:
            return os.path.join(snapshots_dir, subfolders[0])

    return full_path


def save_audio_file(temp_folder, subfolder, text_snippet):
    save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(save_path, exist_ok=True)

    timestamp = datetime.now().strftime("%H-%M-%S")
    clean_text = re.sub(r'[^\w\s-]', '', text_snippet)[:FILENAME_MAX_LEN].strip().replace(' ', '_') or "audio"
    filename = f"{timestamp}_{clean_text}.wav"
    final_path = os.path.join(save_path, filename)

    source_file = os.path.join(temp_folder, "audio_000.wav")

    if os.path.exists(source_file):
        shutil.move(source_file, final_path)
        print(f"Saved: outputs/{subfolder}/{filename}")

        if AUTO_PLAY:
            print("Playing...")
            try:
                subprocess.run(["afplay", final_path], check=False, 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                pass

    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder, ignore_errors=True)


def clean_path(user_input):
    path = user_input.strip()
    if len(path) > 1 and path[0] in ["'", '"'] and path[-1] == path[0]:
        path = path[1:-1]
    return path.replace("\\ ", " ")


def get_safe_input(prompt="\nEnter text (or drag .txt file): "):
    try:
        raw_input = input(prompt).strip()
        if raw_input.lower() in ['exit', 'quit', 'q']:
            return None

        clean_p = clean_path(raw_input)
        if os.path.exists(clean_p) and clean_p.endswith(".txt"):
            print(f"Reading from: {os.path.basename(clean_p)}")
            try:
                with open(clean_p, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except IOError as e:
                print(f"Error reading file: {e}")
                return None

        return raw_input
    except KeyboardInterrupt:
        flush_input()
        return None


def convert_audio_if_needed(input_path):
    if not os.path.exists(input_path):
        return None

    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)

    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, 'rb') as f:
                if f.getnchannels() > 0:
                    return input_path
        except wave.Error:
            pass

    temp_wav = os.path.join(os.getcwd(), f"temp_convert_{int(time.time())}.wav")
    print(f"Converting '{ext}' to WAV...")

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", input_path, 
           "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", temp_wav]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return temp_wav
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Could not convert audio. Is ffmpeg installed?")
        return None


def get_saved_voices():
    if not os.path.exists(VOICES_DIR):
        return []
    voices = [f.replace(".wav", "") for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
    return sorted(voices)


def enroll_new_voice():
    print("\n--- Enroll New Voice ---")
    flush_input()

    name = input("1. Voice name (e.g. Boss, Mom): ").strip()
    if not name:
        return

    safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

    ref_input = input("2. Drag & Drop Reference File: ").strip()
    raw_path = clean_path(ref_input)

    if len(raw_path) > 300 or "\n" in raw_path:
        print("Error: Input too long.")
        flush_input()
        return

    clean_wav_path = convert_audio_if_needed(raw_path)
    if not clean_wav_path:
        return

    print("3. Transcript (important for quality):")
    ref_text = input("   Type EXACTLY what the audio says: ").strip()

    if not os.path.exists(VOICES_DIR):
        os.makedirs(VOICES_DIR)

    target_wav = os.path.join(VOICES_DIR, f"{safe_name}.wav")
    target_txt = os.path.join(VOICES_DIR, f"{safe_name}.txt")

    shutil.copy(clean_wav_path, target_wav)
    with open(target_txt, "w", encoding='utf-8') as f:
        f.write(ref_text)

    if clean_wav_path != raw_path and os.path.exists(clean_wav_path):
        os.remove(clean_wav_path)

    print(f"Voice saved as '{safe_name}'")


def run_custom_session(model_key):
    info = MODELS[model_key]
    model_path = get_smart_path(info["folder"])
    if not model_path:
        print("Error: Model not found.")
        return

    print(f"\nLoading {info['name']}...")
    try:
        model = load_model(model_path)
        model = optimize_model(model)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print(f"\n--- {info['name']} ---")
    speaker = "Vivian"
    all_speakers = [n for names in SPEAKER_MAP.values() for n in names]
    print("Available Speakers: " + ", ".join(all_speakers))

    user_choice = input("\nSelect Speaker (Name): ").strip()
    for lang, names in SPEAKER_MAP.items():
        if user_choice in names:
            speaker = user_choice
            break
    print(f"Using: {speaker}")

    print("\nEmotion Examples:")
    for ex in EMOTION_EXAMPLES:
        print(f"  - {ex}")
    base_instruct = input("Emotion Instruction: ").strip() or "Normal tone"

    print("\nSpeed:")
    print("  1. Normal (1.0x)")
    print("  2. Fast (1.3x)")
    print("  3. Slow (0.8x)")
    sp = input("Choice (1-3): ").strip()
    speed = 1.0
    if sp == "2":
        speed = 1.3
    elif sp == "3":
        speed = 0.8

    while True:
        text = get_safe_input()
        if text is None:
            break
        print("Generating...")
        temp_dir = make_temp_dir()
        try:
            generate_audio(model=model, text=text, voice=speaker, 
                         instruct=base_instruct, speed=speed, output_path=temp_dir)
            save_audio_file(temp_dir, info["output_subfolder"], text)
        except Exception as e:
            print(f"Error: {e}")
    clean_memory()


def run_design_session(model_key):
    info = MODELS[model_key]
    model_path = get_smart_path(info["folder"])
    if not model_path:
        print("Error: Model not found.")
        return

    print(f"\nLoading {info['name']}...")
    try:
        model = load_model(model_path)
        model = optimize_model(model)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    print(f"\n--- {info['name']} ---")
    instruct = input("Describe the voice: ").strip()
    if not instruct:
        return

    while True:
        text = get_safe_input()
        if text is None:
            break
        print("Generating...")
        temp_dir = make_temp_dir()
        try:
            generate_audio(model=model, text=text, instruct=instruct, output_path=temp_dir)
            save_audio_file(temp_dir, info["output_subfolder"], text)
        except Exception as e:
            print(f"Error: {e}")
    clean_memory()


def run_clone_manager(model_key):
    print("\n--- Voice Cloning Manager ---")
    print("  1. Pick from Saved Voices")
    print("  2. Enroll New Voice")
    print("  3. Quick Clone")
    print("  4. Back")

    sub_choice = input("\nChoice: ").strip()
    if sub_choice == "2":
        enroll_new_voice()
        return
    if sub_choice == "4":
        return

    info = MODELS[model_key]
    model_path = get_smart_path(info["folder"])
    if not model_path:
        print("Error: Model not found.")
        return

    print("\nLoading Base Model...")
    try:
        model = load_model(model_path)
        model = optimize_model(model)
    except Exception as e:
        print(f"Load failed: {e}")
        return

    ref_audio, ref_text = None, None

    if sub_choice == "1":
        saved = get_saved_voices()
        if not saved:
            print("No saved voices found.")
            return
        print("\nSaved Voices:")
        for i, v in enumerate(saved):
            print(f"  {i+1}. {v}")
        try:
            idx = int(input("\nPick Number: ")) - 1
            if idx < 0 or idx >= len(saved):
                print("Invalid selection.")
                return
            name = saved[idx]
            ref_audio = os.path.join(VOICES_DIR, f"{name}.wav")
            txt_path = os.path.join(VOICES_DIR, f"{name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
            print(f"Loaded: {name}")
        except (ValueError, IndexError):
            print("Invalid selection.")
            return

    elif sub_choice == "3":
        ref_input = input("\nDrag Reference Audio: ").strip()
        raw_path = clean_path(ref_input)
        ref_audio = convert_audio_if_needed(raw_path)
        if not ref_audio:
            return
        ref_text = input("   Transcript (Optional): ").strip() or "."

    else:
        return

    while True:
        text = get_safe_input(f"\nText for '{os.path.basename(str(ref_audio))}' (or 'exit'): ")
        if text is None:
            break
        print("Cloning...")
        temp_dir = make_temp_dir()
        try:
            generate_audio(model=model, text=text, ref_audio=ref_audio, 
                         ref_text=ref_text, output_path=temp_dir)
            save_audio_file(temp_dir, info["output_subfolder"], text)
        except Exception as e:
            print(f"Error: {e}")
    clean_memory()


def main_menu():
    print("\n" + "=" * 40)
    print(" Qwen3-TTS Manager")
    print("=" * 40)
    
    print("\n  Pro Models (1.7B - Best Quality)")
    print("  ---------------------------------")
    print("  1. Custom Voice")
    print("  2. Voice Design")
    print("  3. Voice Cloning")
    
    print("\n  Lite Models (0.6B - Faster)")
    print("  ---------------------------")
    print("  4. Custom Voice")
    print("  5. Voice Design")
    print("  6. Voice Cloning")

    print("\n  Pro 4-bit (1.7B - Faster, Slightly Lower Quality)")
    print("  ---------------------------")
    print("  7. Custom Voice (4-bit)")

    print("\n  q. Exit")

    choice = input("\nSelect: ").strip().lower()

    if choice == "q":
        sys.exit()

    if choice not in MODELS:
        print("Invalid selection.")
        flush_input()
        return

    mode = MODELS[choice]["mode"]

    if mode == "custom":
        run_custom_session(choice)
    elif mode == "design":
        run_design_session(choice)
    elif mode == "clone_manager":
        run_clone_manager(choice)


if __name__ == "__main__":
    try:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        while True:
            main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
