import uvicorn
import socketio
import asyncio
import sys
import os
import base64
import numpy as np
import cv2
from typing import Dict, Any, Optional
import base64
import uuid
import asyncio
import librosa
import torch

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
PORT = int(os.getenv("PORT", 3000))
HEALTH_KEYWORDS = ["병원", "진료", "의사", "간호사", "증상", "아파", "예약", "상담", "건강", "수술", "검진", "약", "복용"]

# ----------------------------------------------------------------
# 1. 전역 모델 로딩 (Singleton)
# ----------------------------------------------------------------
# 서버 시작 시 딱 한 번만 무거운 모델들을 로딩합니다.
global_models = ModelContainer.get_instance()

# ----------------------------------------------------------------
# 2. 서버 및 세션 설정
# ----------------------------------------------------------------
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

sessions: Dict[str, Router] = {}

# ----------------------------------------------------------------
# 3. 헬퍼 함수 (중복 로직 제거)
# ----------------------------------------------------------------
def format_response_payload(response_data: Any) -> Dict[str, Any]:
    """
    Router의 반환값을 분석하여 클라이언트 규격(JSON)으로 변환합니다.
    """
    data, meta = None, None

    # (데이터, 메타데이터) 튜플 형태인지 확인
    if isinstance(response_data, (tuple, list)) and len(response_data) == 2:
        data, meta = response_data
    else:
        data = response_data

    # set 타입은 JSON 직렬화 불가하므로 list로 변환
    if isinstance(data, set):
        data = list(data)

    # 메타데이터 유무에 따라 응답 타입 결정
    msg_type = "confirm" if meta else "simple"
    
    return {
        "text": data,
        "type": msg_type,
        "meta": meta # 필요하다면 메타데이터도 함께 전송
    }

def decode_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Base64 문자열을 OpenCV 이미지 객체로 변환합니다.
    """
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
    """
    세션을 안전하게 가져오거나, 없으면 재생성합니다.
    """
    if sid not in sessions:
        try:
            # [중요] 재생성 시에도 반드시 전역 모델을 주입해야 합니다.
            sessions[sid] = Router(models=global_models)
        except Exception as e:
            print(f"🚨 Router 재생성 실패 ({sid}): {e}")
            return None
    return sessions[sid]


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
    sessions.pop(sid, None) # 안전한 삭제

@sio.on('command')
async def handle_command(sid, data):
    """ 앱 -> 서버: 텍스트 메시지 전송 """
    print(f"📩 수신 ({sid}): {data}")
    user_text = data.get('text', '')
    
    router = get_or_create_router(sid)
    if not router:
        await sio.emit('command-response', {"text": "서버 초기화 오류", "type": "error"}, to=sid)
        return

    try:
        # AI 추론 (비동기 스레드 실행)
        response_data = await asyncio.to_thread(router.handle, user_text)
        
        # 응답 포맷팅 및 전송
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)
        print(f"📤 전송: {payload}")

    except Exception as e:
        print(f"🚨 처리 중 에러: {e}")
        await sio.emit('command-response', {"text": "처리 중 오류 발생", "type": "error"}, to=sid)


@sio.on('action-confirm')
async def handle_action_confirm(sid, data):
    """ 앱 -> 서버: [네] 버튼 클릭 """
    print(f"🔘 버튼 클릭 수신 (YES): {data}")
    
    router = get_or_create_router(sid)
    if not router:
        return

    try:
        # Decision 로직 수행 ("네"라는 텍스트로 처리)
        response_data = await asyncio.to_thread(router.handle, "네")
        
        # 응답 포맷팅 및 전송 (handle_command와 동일 로직 사용)
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)
        print(f"📤 실행 완료 응답 전송: {payload}")

    except Exception as e:
        print(f"🚨 실행 중 에러: {e}")
        await sio.emit('command-response', {"text": "실행 오류 발생", "type": "error"}, to=sid)

@sio.on('audio-upload')
async def handle_audio_upload(sid, data):
    """
    앱 -> 서버: 수신 -> 전처리 -> [하이브리드 STT] -> Router -> 응답
    """
    print(f"🎤 오디오 데이터 수신 ({sid})")
    
    router = get_or_create_router(sid)
    if not router:
        return

    raw_filename = None
    processed_filename = None

    try:
        # -------------------------------------------------------
        # 1. 데이터 파싱 및 파일 저장
        # -------------------------------------------------------
        b64_string = data.get('audioData')
        file_ext = data.get('format', 'm4a')
        user_id = data.get('userId', 'unknown')

        audio_bytes = base64.b64decode(b64_string)
        
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
            
        raw_filename = f"uploads/{user_id}_{uuid.uuid4()}.{file_ext}"
        with open(raw_filename, "wb") as f:
            f.write(audio_bytes)
            
        print(f"💾 원본 저장 완료: {raw_filename}")

        # -------------------------------------------------------
        # 2. 오디오 전처리 (Preprocessing)
        # -------------------------------------------------------
        def preprocess_audio():
            print("🎛️ 전처리: Resample(16k) & Normalize")
            audio = AudioSegment.from_file(raw_filename, format=file_ext)
            audio = audio.set_channels(1)       # Mono
            audio = audio.set_frame_rate(16000) # 16kHz
            audio = pydub_normalize(audio)      # Volume Maximize
            
            new_filename = raw_filename.replace(f".{file_ext}", "_processed.wav")
            audio.export(new_filename, format="wav")
            return new_filename

        processed_filename = await asyncio.to_thread(preprocess_audio)
        print(f"✨ 전처리 완료: {processed_filename}")

        # -------------------------------------------------------
        # 3. [1차] 일반 Whisper 인식 (확신도 체크)
        # -------------------------------------------------------
        print("👂 [1단계] 일반 모델 인식 중...")
        models = global_models # ModelContainer 인스턴스

        def transcribe_std():
            # OpenAI 모델은 딕셔너리를 반환하며 'segments' 안에 'avg_logprob'가 있음
            result = models.stt_model.transcribe(
                processed_filename, 
                language="ko", 
                fp16=False,
                beam_size=5,
                initial_prompt="건강 상담, 몸 상태, 허약 체질, 병원 진료에 대한 대화입니다."
            )
            
            text = result['text'].strip()
            # 확신도(Log Probability) 추출 (0에 가까울수록 확실, -1 이하면 불확실)
            # segments가 비어있을 수 있으므로 예외처리
            score = -10.0
            if result.get('segments'):
                score = result['segments'][0].get('avg_logprob', -10.0)
            
            return text, score

        text_std, score_std = await asyncio.to_thread(transcribe_std)
        print(f"🗣️ [1차 결과] '{text_std}' (확신도: {score_std:.2f})")

        # -------------------------------------------------------
        # 4. [판단] 구음장애 모델 가동 여부 결정
        # -------------------------------------------------------
        use_dys_model = False
        
        # 조건 A: 확신도가 낮음 (AI가 잘 못알아들음) -> -0.7 기준 (조정 가능)
        if score_std < -0.7:
            use_dys_model = True
            print("📉 확신도 낮음 -> 특화 모델 전환")
            
        # 조건 B: 텍스트가 너무 짧음 (오인식 가능성 높음)
        elif len(text_std) < 3:
            use_dys_model = True
            print("📉 텍스트 너무 짧음 -> 특화 모델 전환")

        # 조건 C: [Override] 병원 키워드가 있으면 무조건 일반 모델 신뢰
        # (구음장애 모델은 '비타민', '연필' 등 생활 용어에 편향되어 있을 수 있음)
        for kw in HEALTH_KEYWORDS:
            if kw in text_std:
                use_dys_model = False
                print(f"🏥 병원 키워드('{kw}') 감지 -> 일반 모델 유지")
                break

        final_text = text_std

        # -------------------------------------------------------
        # 5. [2차] 구음장애 특화 모델 (필요 시 실행)
        # -------------------------------------------------------
        if use_dys_model:
            print("🚀 [2단계] 구음장애 특화 모델 가동")

            def transcribe_dys():
                # Librosa로 로드 (WhisperProcessor 입력용)
                audio_array, _ = librosa.load(processed_filename, sr=16000)
                
                # Processor 전처리
                inputs = models.dys_processor(
                    audio_array, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_features.to(models.device)

                # 추론
                with torch.no_grad():
                    generated_ids = models.dys_model.generate(inputs, language="korean")

                # 디코딩
                transcription = models.dys_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return transcription.strip()

            text_dys = await asyncio.to_thread(transcribe_dys)
            print(f"🗣️ [2차 결과] '{text_dys}'")

            # 특화 모델 결과가 유의미하면 채택
            if text_dys:
                final_text = text_dys

        # -------------------------------------------------------
        # 6. [후처리] 텍스트 교정 (Post-processing)
        # -------------------------------------------------------
        # "화약" -> "허약" 강제 치환
        if "화약" in final_text:
            final_text = final_text.replace("화약", "허약")
            print("🔧 텍스트 교정: 화약 -> 허약")

        print(f"✅ 최종 확정: \"{final_text}\"")

        # -------------------------------------------------------
        # 7. 응답 처리
        # -------------------------------------------------------
        if not final_text:
            await sio.emit('command-response', {"text": "잘 듣지 못했어요. 다시 말씀해 주세요.", "type": "simple"}, to=sid)
            return

        # 앱에 내 말 먼저 띄우기
        await sio.emit('user-speech', {'text': final_text}, to=sid)

        # Router 실행
        response_data = await asyncio.to_thread(router.handle, final_text)
        
        # 최종 응답 전송
        payload = format_response_payload(response_data)
        await sio.emit('command-response', payload, to=sid)
        print(f"📤 응답 전송: {payload}")

    except Exception as e:
        print(f"🚨 오디오 처리 중 에러: {e}")
        import traceback
        traceback.print_exc() # 상세 에러 로그 출력
        await sio.emit('command-response', {"text": "처리 중 오류가 발생했습니다.", "type": "error"}, to=sid)
    
    finally:
        # 파일 정리
        try:
            if raw_filename and os.path.exists(raw_filename):
                os.remove(raw_filename)
            if processed_filename and os.path.exists(processed_filename):
                os.remove(processed_filename)
        except Exception:
            pass

@sio.on('identify-face')
async def handle_identify_face(sid, base64_image):
    """ Expo 앱의 'identify-face' 이벤트 """
    print(f"📸 {sid} 이미지 수신 ({len(base64_image)} bytes)")

    try:
        # 이미지 디코딩 헬퍼 사용
        img = await asyncio.to_thread(decode_image, base64_image)
        
        if img is None:
            await sio.emit('auth-fail', {"reason": "image_decode_error"}, to=sid)
            return

        # ... (얼굴 인식 로직 시뮬레이션) ...
        # 실제로는 여기서 img 변수를 face_recognition 모델에 넘깁니다.
        await asyncio.sleep(0.5) 
        user = {"id": "p123", "name": "김블라"}

        await sio.emit('auth-success', user, to=sid)
        print(f"✅ 인증 성공: {user['name']}")

    except Exception as e:
        print(f"🚨 인증 처리 중 오류: {e}")
        await sio.emit('auth-fail', to=sid)

if __name__ == "__main__":
    print(f"🚀 AI Router 서버 시작 (Port: {PORT})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)