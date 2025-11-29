import uvicorn
import socketio
import asyncio
import sys
import os
import base64
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional
import uuid
import librosa 
import torch
from contextlib import nullcontext
import tempfile

from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize

# 커스텀 모듈
from core.router import Router
from core.model_loader import ModelContainer

try:
    import audioop_lts
    sys.modules["audioop"] = audioop_lts
except ImportError:
    pass

# 환경 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PORT = int(os.getenv("PORT", 3000))

# [수정] 다국어 키워드 및 시스템 메시지 정의
KEYWORDS = {
    "ko": {
        "health": ["병원", "진료", "의사", "간호사", "증상", "아파", "예약", "상담", "건강", "수술", "검진", "약", "복용"],
        "emergency": ["정지", "멈춰", "서라", "스톱", "STOP", "SOS", "비상"],
        "whisper_lang": "korean"
    },
    "en": {
        "health": ["hospital", "doctor", "nurse", "symptom", "pain", "hurt", "appointment", "consult", "health", "surgery", "checkup", "medicine", "pill"],
        "emergency": ["stop", "halt", "freeze", "emergency", "sos", "help"],
        "whisper_lang": "english"
    }
}

SYSTEM_MESSAGES = {
    "ko": {
        "emergency_stop": "비상 정지합니다.",
        "emergency_ack": "비상 정지 명령을 확인했습니다. 로봇을 즉시 정지합니다.",
        "server_error": "서버 초기화 오류",
        "process_error": "처리 중 오류 발생",
        "exec_error": "실행 오류 발생",
        "not_heard": "잘 듣지 못했어요.",
        "decode_error": "이미지 디코딩 실패"
    },
    "en": {
        "emergency_stop": "Initiating emergency stop.",
        "emergency_ack": "Emergency stop command received. Stopping robot immediately.",
        "server_error": "Server initialization error",
        "process_error": "Processing error",
        "exec_error": "Execution error",
        "not_heard": "I couldn't hear you clearly.",
        "decode_error": "Image decode failed"
    }
}

# ----------------------------------------------------------------
# 성능 측정 유틸리티 클래스
# ----------------------------------------------------------------
class PerformanceTimer:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (time.perf_counter() - self.start_time) * 1000
        status = "❌ 실패" if exc_type else "✅ 완료"
        print(f"⏱️ [Perf] [{self.task_name}] {status} | 소요시간: {elapsed:.2f}ms")

# ----------------------------------------------------------------
# 1. 전역 모델 로딩 (Singleton)
# ----------------------------------------------------------------
global_models = ModelContainer.get_instance()

# ----------------------------------------------------------------
# 2. 서버 및 세션 설정
# ----------------------------------------------------------------
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

sessions: Dict[str, Router] = {}

# ----------------------------------------------------------------
# 3. 헬퍼 함수
# ----------------------------------------------------------------
def format_response_payload(response_data: Any) -> Dict[str, Any]:
    data, meta = None, None
    if isinstance(response_data, (tuple, list)) and len(response_data) == 2:
        data, meta = response_data
    else:
        data = response_data

    if isinstance(data, set):
        data = list(data)

    msg_type = "confirm" if meta else "simple"
    return {"text": data, "type": msg_type, "meta": meta}

def decode_image(base64_string: str) -> Optional[np.ndarray]:
    try:
        if ',' in base64_string:
            _, base64_data = base64_string.split(',', 1)
        else:
            base64_data = base64_string
        img_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"🖼 이미지 디코딩 실패: {e}")
        return None

def get_or_create_router(sid: str) -> Optional[Router]:
    if sid not in sessions:
        try:
            with PerformanceTimer(f"Router 생성 ({sid})"):
                sessions[sid] = Router(models=global_models)
        except Exception as e:
            print(f"🚨 Router 재생성 실패 ({sid}): {e}")
            return None
    return sessions[sid]

# [수정] 언어별 메시지 처리
async def execute_emergency_stop(sid, user_id, lang="ko"):
    print(f"🛑 [EMERGENCY] 로봇 정지 실행: User={user_id}, Lang={lang}")
    msg = SYSTEM_MESSAGES.get(lang, SYSTEM_MESSAGES["ko"])["emergency_stop"]
    
    await sio.emit('command-response', {
        "text": msg, 
        "type": "simple",
        "meta": {"emergency": True}
    }, to=sid)

def build_prompt_with_lang(text: str, lang: str) -> str:
    """언어 설정에 따라 Router에 전달할 텍스트 가공"""
    if lang == 'en':
        return f"{text} (Please respond in English)"
    return text

# ----------------------------------------------------------------
# 4. 이벤트 핸들러
# ----------------------------------------------------------------

@sio.event
async def connect(sid, environ):
    print(f"✅ 클라이언트 연결됨: {sid}")
    get_or_create_router(sid)

@sio.event
async def disconnect(sid):
    print(f"❌ 클라이언트 연결 끊김: {sid}")
    sessions.pop(sid, None)

@sio.on('pause')
async def handle_pause(sid, data):
    print(f"\n[!!! EMERGENCY !!!] 🚨 비상 정지 요청됨 ({data})")
    router = get_or_create_router(sid)
    
    user_text = data.get('text', '')
    lang = data.get('lang', 'ko') 
    
    sys_msg = SYSTEM_MESSAGES.get(lang, SYSTEM_MESSAGES["ko"])

    with PerformanceTimer("비상 정지 처리"):
        # Router는 Pause 처리에 언어가 필요 없을 수 있지만, 혹시 모르니 전달 로직 유지
        await asyncio.to_thread(router.handle, user_text)
    
    await sio.emit('command-response', {
        "text": sys_msg["emergency_ack"],
        "type": "simple",
        "meta": {"status": "stopped", "emergency": True}
    }, to=sid)

@sio.on('command')
async def handle_command(sid, data):
    print(f"📩 수신 ({sid}): {data}")
    user_text = data.get('text', '')
    lang = data.get('lang', 'ko') 
    sys_msg = SYSTEM_MESSAGES.get(lang, SYSTEM_MESSAGES["ko"])

    router = get_or_create_router(sid)
    if not router:
        await sio.emit('command-response', {"text": sys_msg["server_error"], "type": "error"}, to=sid)
        return

    try:
        with PerformanceTimer("텍스트 명령 처리 (Router)"):
            # [핵심 수정] 언어가 영어일 경우 Router에게 지시어 전달
            router_input = user_text
            if lang == 'en':
                router_input = f"{user_text} (Please respond in English)"
            
            response_data = await asyncio.to_thread(router.handle, router_input)
        
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)
        print(f"📤 전송: {payload}")
    except Exception as e:
        print(f"🚨 처리 중 에러: {e}")
        await sio.emit('command-response', {"text": sys_msg["process_error"], "type": "error"}, to=sid)

@sio.on('action-confirm')
async def handle_action_confirm(sid, data):
    print(f"🔘 버튼 클릭 수신 (YES): {data}")
    lang = data.get('lang', 'ko')
    sys_msg = SYSTEM_MESSAGES.get(lang, SYSTEM_MESSAGES["ko"])

    router = get_or_create_router(sid)
    if not router: return
    try:
        with PerformanceTimer("확인 명령 처리 (Router)"):
            # [핵심 수정] 긍정 답변도 언어에 맞게 변환 및 지시어 추가
            confirm_text = "Yes" if lang == "en" else "네"
            if lang == 'en':
                confirm_text += " (Please respond in English)"
            
            response_data = await asyncio.to_thread(router.handle, confirm_text)
        
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)
    except Exception as e:
        print(f"🚨 실행 중 에러: {e}")
        await sio.emit('command-response', {"text": sys_msg["exec_error"], "type": "error"}, to=sid)

@sio.on('audio-upload')
async def handle_audio_upload(sid, data):
    print(f"🎤 오디오 데이터 수신 ({sid})")
    total_timer = PerformanceTimer("오디오 처리 전체 (Total Flow)")
    total_timer.__enter__()

    # 1. 언어 및 설정 로드
    lang = data.get('lang', 'ko')
    if lang not in KEYWORDS: lang = "ko"
    
    current_keywords = KEYWORDS[lang]
    sys_msg = SYSTEM_MESSAGES[lang]
    # HuggingFace Whisper는 'korean', 'english' 등으로 풀네임 사용 권장
    whisper_lang_code = current_keywords["whisper_lang"] 

    router = get_or_create_router(sid)
    if not router: 
        total_timer.__exit__(None, None, None)
        return

    # 임시 파일 경로 변수 초기화
    temp_raw_path = None
    temp_wav_path = None

    # 전역 모델 컨테이너 참조
    models = global_models 

    try:
        # ------------------------------------------------------------------
        # 1. 파일 저장 및 전처리 (Tempfile 사용으로 안정성 확보)
        # ------------------------------------------------------------------
        with PerformanceTimer("1. 오디오 파일 저장 및 변환"):
            b64_string = data.get('audioData')
            file_ext = data.get('format', 'm4a')
            
            # Base64 디코딩
            try:
                if not b64_string: raise ValueError("Empty audio data")
                audio_bytes = base64.b64decode(b64_string)
            except Exception:
                print("🚨 Base64 디코딩 실패")
                await sio.emit('command-response', {"text": sys_msg["process_error"], "type": "error"}, to=sid)
                return

            # (1) Raw 파일 생성 (자동 삭제 방지를 위해 delete=False, finally에서 삭제)
            with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp_raw:
                tmp_raw.write(audio_bytes)
                temp_raw_path = tmp_raw.name
            
            # (2) WAV 변환 대상 경로 생성
            temp_wav_path = temp_raw_path.replace(f".{file_ext}", "_processed.wav")

            # (3) Pydub 변환 (블로킹 작업이므로 스레드 분리)
            def convert_audio():
                audio = AudioSegment.from_file(temp_raw_path, format=file_ext)
                audio = audio.set_channels(1)       # 모노
                audio = audio.set_frame_rate(16000) # 16kHz (Whisper 표준)
                audio = pydub_normalize(audio)      # 볼륨 정규화
                audio.export(temp_wav_path, format="wav")
                return temp_wav_path

            await asyncio.to_thread(convert_audio)

        # ------------------------------------------------------------------
        # 2. [1차] Medium 모델로 일반 인식 (순정 모델 사용)
        # ------------------------------------------------------------------
        print(f"👂 [1단계] Medium 모델 인식 (Lang: {whisper_lang_code})...")

        def transcribe_medium():
            # librosa로 로드 (sr=16000)
            audio_array, _ = librosa.load(temp_wav_path, sr=16000) 
            
            # Processor는 공용 사용 (토크나이저 호환됨)
            inputs = models.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(models.device)
            
            # [1차] Medium 모델 추론 (순정)
            # stt_model_medium은 순정 WhisperForConditionalGeneration 객체임
            with torch.no_grad():
                outputs = models.stt_model_medium.generate(
                    inputs,
                    language=whisper_lang_code, 
                    max_new_tokens=128,
                    return_dict_in_generate=True, 
                    output_scores=True            
                )
            
            generated_ids = outputs.sequences
            text = models.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 신뢰도(Log Probability) 계산
            transition_scores = models.stt_model_medium.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            avg_logprob = torch.mean(transition_scores[0]).item()
            
            return text.strip(), avg_logprob

        with PerformanceTimer("3. STT 1차 (Whisper Medium)"):
            text_std, score_std = await asyncio.to_thread(transcribe_medium)
        
        print(f"🗣️ [1차 Medium 결과] '{text_std}' (신뢰도: {score_std:.4f})")

        # ------------------------------------------------------------------
        # 3. [판단] 적합도 검사 및 분기
        # ------------------------------------------------------------------
        use_dys_model = False
        # Medium은 성능이 좋으므로 임계값을 조금 낮게 잡아도 됨 (예: -0.6 ~ -0.7)
        CONFIDENCE_THRESHOLD = -0.6 

        if score_std < CONFIDENCE_THRESHOLD:
            print(f"📉 신뢰도 낮음({score_std:.2f}) -> 2차 검증 필요")
            use_dys_model = True
        elif len(text_std) < 2: 
            print("📉 텍스트 너무 짧음 -> 2차 검증 필요")
            use_dys_model = True

        # 중요 키워드가 1차에서 이미 명확히 들렸다면 2차 생략 (오탐 방지)
        for kw in current_keywords["health"]:
            if kw.lower() in text_std.lower(): use_dys_model = False; break
        for kw in current_keywords["emergency"]:
            if kw.lower() in text_std.lower(): use_dys_model = False; break

        final_text = text_std

        # ------------------------------------------------------------------
        # 4. [2차] Small + LoRA 모델로 정밀 인식 (한국어일 때만 수행)
        # ------------------------------------------------------------------
        if use_dys_model and lang == 'ko': 
            print("🚀 [2단계] Small + LoRA 모델 가동 (Beam Search)")

            def transcribe_small_lora():
                audio_array, _ = librosa.load(temp_wav_path, sr=16000)
                inputs = models.processor(
                    audio_array, sampling_rate=16000, return_tensors="pt"
                ).input_features.to(models.device)

                # [2차] Small + LoRA 모델 추론
                # stt_model_small_lora는 PeftModel 객체임
                with torch.no_grad():
                    generated_ids = models.stt_model_small_lora.generate(
                        inputs, 
                        language=whisper_lang_code,
                        num_beams=5,             # 빔 서치로 정확도 향상
                        num_return_sequences=3,  # 상위 3개 후보 추출
                        early_stopping=True
                    )
                return models.processor.batch_decode(generated_ids, skip_special_tokens=True)

            with PerformanceTimer("4. STT 2차 (Small+LoRA)"):
                candidates = await asyncio.to_thread(transcribe_small_lora)
            
            print(f"🧐 [2차 후보군]: {candidates}")
            
            if candidates:
                final_text = candidates[0] # 기본적으로 1순위 채택
                
                # 후보군 중 응급/건강 키워드가 포함된 문장이 있다면 우선 채택 (Rescue Logic)
                for cand in candidates:
                    all_keywords = current_keywords["emergency"] + current_keywords["health"]
                    if any(kw.lower() in cand.lower() for kw in all_keywords):
                        print(f"✅ 키워드 매칭으로 후보 교체: {cand}")
                        final_text = cand
                        break

        print(f"✅ 최종 확정: \"{final_text}\"")

        # ------------------------------------------------------------------
        # 5. 결과 처리 및 라우터 전달
        # ------------------------------------------------------------------
        
        # 인식된 텍스트가 없으면 종료
        if not final_text:
            await sio.emit('command-response', {"text": sys_msg["not_heard"], "type": "simple"}, to=sid)
            return

        # 사용자에게 인식된 텍스트 전송 (채팅창 표시용)
        await sio.emit('user-speech', {'text': final_text}, to=sid)

        # 비상 정지 키워드 체크 (최우선 순위)
        if any(kw.lower() in final_text.lower() for kw in current_keywords["emergency"]):
            await execute_emergency_stop(sid, data.get('userId', 'unknown'), lang)
            return

        # Router 명령 처리
        with PerformanceTimer("5. Router 명령 처리"):
            # 언어에 따라 프롬프트 조정 (헬퍼 함수 사용)
            router_input = build_prompt_with_lang(final_text, lang)
            
            response_data = await asyncio.to_thread(router.handle, router_input)
        
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)

    except Exception as e:
        print(f"🚨 오디오 처리 중 에러: {e}")
        import traceback; traceback.print_exc()
        await sio.emit('command-response', {"text": sys_msg["process_error"], "type": "error"}, to=sid)
    
    finally:
        # ------------------------------------------------------------------
        # 6. 임시 파일 정리 (반드시 수행)
        # ------------------------------------------------------------------
        try:
            if temp_raw_path and os.path.exists(temp_raw_path):
                os.remove(temp_raw_path)
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            print(f"⚠️ 파일 정리 실패: {e}")
        
        total_timer.__exit__(None, None, None)

@sio.on('identify-face')
async def handle_identify_face(sid, data):
    print(f"📸 {sid} 얼굴 인식 요청 수신")
    
    # [수정] 데이터 파싱 (클라이언트가 객체로 보낼 때와 문자열로 보낼 때 모두 대응)
    lang = "ko"
    base64_image = ""

    if isinstance(data, dict):
        base64_image = data.get('image', '')
        lang = data.get('lang', 'ko')
    else:
        # 기존 클라이언트 호환성 유지 (문자열만 온 경우)
        base64_image = data
        lang = "ko"

    try:
        with PerformanceTimer("이미지 디코딩"):
            img = await asyncio.to_thread(decode_image, base64_image)
        
        if img is None:
            await sio.emit('auth-fail', {"reason": "image_decode_error"}, to=sid)
            return
        
        # [Mock] 얼굴 인식 로직 (가정)
        await asyncio.sleep(0.5)
        
        # [핵심 수정] 언어에 따른 이름 분기 처리
        if lang == 'en':
            user_name = "KimVla"
        else:
            user_name = "김블라"
            
        user = {"id": "p123", "name": user_name}
        
        print(f"✅ 인증 성공: {user_name} ({lang})")
        await sio.emit('auth-success', user, to=sid)
        
    except Exception as e:
        print(f"🚨 인증 오류: {e}")
        await sio.emit('auth-fail', to=sid)

if __name__ == "__main__":
    print(f"🚀 AI Router 서버 시작 (Port: {PORT})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)