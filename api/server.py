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
HEALTH_KEYWORDS = ["병원", "진료", "의사", "간호사", "증상", "아파", "예약", "상담", "건강", "수술", "검진", "약", "복용"]
EMERGENCY_KEYWORDS = ["정지", "멈춰", "서라", "스톱", "STOP", "SOS", "비상"]

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

async def execute_emergency_stop(sid, user_id):
    print(f"🛑 [EMERGENCY] 로봇 정지 실행: User={user_id}")
    await sio.emit('command-response', {
        "text": "비상 정지합니다.", 
        "type": "simple",
        "meta": {"emergency": True}
    }, to=sid)

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
    with PerformanceTimer("비상 정지 처리"):
        await asyncio.to_thread(router.handle, user_text)
    await sio.emit('command-response', {
        "text": "비상 정지 명령을 확인했습니다. 로봇을 즉시 정지합니다.",
        "type": "simple",
        "meta": {"status": "stopped", "emergency": True}
    }, to=sid)

@sio.on('command')
async def handle_command(sid, data):
    print(f"📩 수신 ({sid}): {data}")
    user_text = data.get('text', '')
    router = get_or_create_router(sid)
    if not router:
        await sio.emit('command-response', {"text": "서버 초기화 오류", "type": "error"}, to=sid)
        return

    try:
        with PerformanceTimer("텍스트 명령 처리 (Router)"):
            response_data = await asyncio.to_thread(router.handle, user_text)
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)
        print(f"📤 전송: {payload}")
    except Exception as e:
        print(f"🚨 처리 중 에러: {e}")
        await sio.emit('command-response', {"text": "처리 중 오류 발생", "type": "error"}, to=sid)

@sio.on('action-confirm')
async def handle_action_confirm(sid, data):
    print(f"🔘 버튼 클릭 수신 (YES): {data}")
    router = get_or_create_router(sid)
    if not router: return
    try:
        with PerformanceTimer("확인 명령 처리 (Router)"):
            response_data = await asyncio.to_thread(router.handle, "네")
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)
    except Exception as e:
        print(f"🚨 실행 중 에러: {e}")
        await sio.emit('command-response', {"text": "실행 오류 발생", "type": "error"}, to=sid)

@sio.on('audio-upload')
async def handle_audio_upload(sid, data):
    print(f"🎤 오디오 데이터 수신 ({sid})")
    total_timer = PerformanceTimer("오디오 처리 전체 (Total Flow)")
    total_timer.__enter__()

    router = get_or_create_router(sid)
    if not router: return

    raw_filename = None
    processed_filename = None

    try:
        # 1. 파일 저장
        with PerformanceTimer("1. 오디오 파일 저장"):
            b64_string = data.get('audioData')
            file_ext = data.get('format', 'm4a')
            user_id = data.get('userId', 'unknown')
            audio_bytes = base64.b64decode(b64_string)
            if not os.path.exists('uploads'): os.makedirs('uploads')
            raw_filename = f"uploads/{user_id}_{uuid.uuid4()}.{file_ext}"
            with open(raw_filename, "wb") as f: f.write(audio_bytes)

        # 2. 전처리 (16kHz WAV 변환)
        def preprocess_audio():
            audio = AudioSegment.from_file(raw_filename, format=file_ext)
            audio = audio.set_channels(1)       
            audio = audio.set_frame_rate(16000) 
            audio = pydub_normalize(audio)      
            new_filename = raw_filename.replace(f".{file_ext}", "_processed.wav")
            audio.export(new_filename, format="wav")
            return new_filename

        with PerformanceTimer("2. 오디오 전처리 (Pydub)"):
            processed_filename = await asyncio.to_thread(preprocess_audio)

        # 3. [1차] 일반 인식 (LoRA 비활성화 -> 순정 Whisper)
        print("👂 [1단계] 일반 모델 인식 중 (LoRA Off)...")
        models = global_models 

        def transcribe_std():
            audio_array, _ = librosa.load(processed_filename, sr=16000)
            
            inputs = models.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(models.device)
            
            adapter_context = nullcontext()
            if hasattr(models.stt_model, "disable_adapter"):
                adapter_context = models.stt_model.disable_adapter()

            with adapter_context:
                with torch.no_grad():
                    # [수정] output_scores=True로 신뢰도 점수 계산 준비
                    outputs = models.stt_model.generate(
                        inputs,
                        language="korean",
                        max_new_tokens=128,
                        return_dict_in_generate=True, # 결과 객체 반환
                        output_scores=True            # 점수(Logits) 반환
                    )
            
            # 텍스트 디코딩
            generated_ids = outputs.sequences
            text = models.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # [신규] 신뢰도(Log Prob) 점수 계산
            # transition_scores: 생성된 토큰들의 로그 확률 계산
            transition_scores = models.stt_model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            # 평균 로그 확률 계산 (높을수록 좋음, 보통 0 ~ -N 음수값)
            # exp()를 취하면 확률(0~1)이 되지만, 보통 log_prob 상태로 비교함
            avg_logprob = torch.mean(transition_scores[0]).item()
            
            return text.strip(), avg_logprob

        with PerformanceTimer("3. STT 1차 (Whisper Base)"):
            text_std, score_std = await asyncio.to_thread(transcribe_std)
        
        print(f"🗣️ [1차 결과] '{text_std}' (신뢰도: {score_std:.4f})")

        # 4. [판단] 적합도 검사 (점수가 낮으면 불확실함 -> 구음장애 모델 사용)
        use_dys_model = False
        
        # Whisper의 avg_logprob는 보통 음수입니다. 
        # -1.0 보다 낮으면 확신이 부족한 상태로 간주 (경험적 임계값 -0.7 ~ -1.0)
        # 예: -0.2(매우 확실), -0.8(보통), -1.5(불확실)
        CONFIDENCE_THRESHOLD = -0.5

        if score_std < CONFIDENCE_THRESHOLD:
            print(f"📉 신뢰도 낮음({score_std:.2f} < {CONFIDENCE_THRESHOLD}) -> 정밀 분석 필요")
            use_dys_model = True
        
        # [옵션] 너무 짧은 텍스트도 불확실하므로 포함 (원하시면 제거 가능)
        elif len(text_std) < 2: 
            use_dys_model = True

        # 키워드 체크 (명확한 키워드가 있으면 점수가 낮아도 통과시킬 수 있음)
        for kw in HEALTH_KEYWORDS:
            if kw in text_std: use_dys_model = False; break
        for kw in EMERGENCY_KEYWORDS:
            if kw in text_std: use_dys_model = False; break

        final_text = text_std

        # 5. [2차] 정밀 추론 (LoRA 활성화 + Beam Search)
        if use_dys_model:
            print("🚀 [2단계] 구음장애 특화 모델 가동 (LoRA On + Beam Search)")

            def transcribe_dys_candidates():
                audio_array, _ = librosa.load(processed_filename, sr=16000)
                
                inputs = models.processor(
                    audio_array, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(models.device)

                with torch.no_grad():
                    generated_ids = models.stt_model.generate(
                        inputs, 
                        language="korean",
                        num_beams=5,             
                        num_return_sequences=3,  
                        early_stopping=True
                    )
                
                candidates = models.processor.batch_decode(generated_ids, skip_special_tokens=True)
                return candidates

            with PerformanceTimer("4. STT 2차 (Dysarthria Model)"):
                candidates = await asyncio.to_thread(transcribe_dys_candidates)
            
            print(f"🧐 생성된 후보군: {candidates}")
            
            if candidates:
                final_text = candidates[0]
                for cand in candidates:
                    if any(kw in cand for kw in EMERGENCY_KEYWORDS):
                        final_text = cand
                        break

        print(f"✅ 최종 확정: \"{final_text}\"")

        # 응답 처리
        if not final_text:
            await sio.emit('command-response', {"text": "잘 듣지 못했어요.", "type": "simple"}, to=sid)
            return

        await sio.emit('user-speech', {'text': final_text}, to=sid)

        if any(kw in final_text for kw in EMERGENCY_KEYWORDS):
            await execute_emergency_stop(sid, user_id)
            return

        with PerformanceTimer("5. Router 명령 처리"):
            response_data = await asyncio.to_thread(router.handle, final_text)
        
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)

    except Exception as e:
        print(f"🚨 오디오 처리 중 에러: {e}")
        import traceback; traceback.print_exc()
        await sio.emit('command-response', {"text": "오류 발생", "type": "error"}, to=sid)
    
    finally:
        try:
            if raw_filename and os.path.exists(raw_filename): os.remove(raw_filename)
            if processed_filename and os.path.exists(processed_filename): os.remove(processed_filename)
        except Exception: pass
        total_timer.__exit__(None, None, None)

@sio.on('identify-face')
async def handle_identify_face(sid, base64_image):
    print(f"📸 {sid} 이미지 수신")
    try:
        with PerformanceTimer("이미지 디코딩"):
            img = await asyncio.to_thread(decode_image, base64_image)
        if img is None:
            await sio.emit('auth-fail', {"reason": "image_decode_error"}, to=sid)
            return
        
        await asyncio.sleep(0.5)
        user = {"id": "p123", "name": "김블라"}
        await sio.emit('auth-success', user, to=sid)
    except Exception as e:
        print(f"🚨 인증 오류: {e}")
        await sio.emit('auth-fail', to=sid)

if __name__ == "__main__":
    print(f"🚀 AI Router 서버 시작 (Port: {PORT})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)