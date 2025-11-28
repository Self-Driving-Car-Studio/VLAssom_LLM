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
    앱 -> 서버: 음성 수신 -> [전처리] -> Whisper STT -> Router -> 응답
    """
    print(f"🎤 오디오 데이터 수신 ({sid})")
    
    router = get_or_create_router(sid)
    if not router:
        return

    # 임시 파일 경로 변수들 초기화
    raw_filename = None
    processed_filename = None

    try:
        # 1. 데이터 파싱
        b64_string = data.get('audioData')
        file_ext = data.get('format', 'm4a')
        user_id = data.get('userId', 'unknown')

        # Base64 디코딩
        audio_bytes = base64.b64decode(b64_string)
        
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
            
        # 2. 원본 파일 저장 (.m4a)
        raw_filename = f"uploads/{user_id}_{uuid.uuid4()}.{file_ext}"
        with open(raw_filename, "wb") as f:
            f.write(audio_bytes)
            
        print(f"💾 원본 저장 완료: {raw_filename}")

        # =======================================================
        # [✨ 추가됨] 3. 오디오 전처리 (Preprocessing)
        # Whisper가 가장 좋아하는 형태(16kHz, Mono, Normalized)로 변환
        # =======================================================
        def preprocess_audio():
            print("🎛️ 오디오 전처리 중... (Resample & Normalize)")
            
            # 원본 로드
            audio = AudioSegment.from_file(raw_filename, format=file_ext)
            
            # (1) 모노로 변환 (채널 1개)
            audio = audio.set_channels(1)
            
            # (2) 주파수 16000Hz로 변경 (Whisper 내부 표준)
            audio = audio.set_frame_rate(16000)
            
            # (3) 볼륨 정규화 (작은 목소리 증폭)
            audio = pydub_normalize(audio)
            
            # 전처리된 파일명 생성 (.wav)
            new_filename = raw_filename.replace(f".{file_ext}", "_processed.wav")
            
            # wav 포맷으로 저장
            audio.export(new_filename, format="wav")
            return new_filename

        # 전처리 실행 (동기 작업이므로 스레드로 분리 권장)
        processed_filename = await asyncio.to_thread(preprocess_audio)
        print(f"✨ 전처리 완료: {processed_filename}")

        # =======================================================
        # 4. Whisper STT 변환
        # =======================================================
        print("👂 Whisper 인식 중...")
        stt_model = global_models.stt_model
        
        def transcribe_audio():
            # [중요] 원본 대신 '전처리된 wav 파일'을 넣습니다.
            # beam_size=5: 정확도를 위해 탐색 폭을 넓힘 (기본값 1보다 느리지만 정확함)
            return stt_model.transcribe(
                processed_filename, 
                language="ko", 
                fp16=False,
                beam_size=5,
                initial_prompt="건강 상담, 몸 상태, 허약 체질, 병원 진료에 대한 대화입니다."
            )

        result = await asyncio.to_thread(transcribe_audio)
        recognized_text = result['text'].strip()
        
        print(f"🗣️ 인식된 텍스트: \"{recognized_text}\"")

        # -------------------------------------------------------
        # 5. 실패 처리 및 사용자 피드백 전송
        # -------------------------------------------------------
        if not recognized_text:
            await sio.emit('command-response', {"text": "음성이 너무 작거나 들리지 않았어요.", "type": "simple"}, to=sid)
        else:
            # 인식 성공 시, 앱에 내 말 먼저 띄워주기
            await sio.emit('user-speech', {'text': recognized_text}, to=sid)

            # 6. Router 실행
            response_data = await asyncio.to_thread(router.handle, recognized_text)
            
            # 7. 최종 응답
            payload = format_response_payload(response_data)
            await sio.emit('command-response', payload, to=sid)
            print(f"📤 응답 전송: {payload}")

    except Exception as e:
        print(f"🚨 오디오 처리 중 에러: {e}")
        await sio.emit('command-response', {"text": "오류가 발생했습니다.", "type": "error"}, to=sid)
    
    finally:
        # 8. [청소] 임시 파일들 삭제 (용량 관리)
        try:
            if raw_filename and os.path.exists(raw_filename):
                os.remove(raw_filename)
            if processed_filename and os.path.exists(processed_filename):
                os.remove(processed_filename)
        except Exception as cleanup_error:
            print(f"🧹 파일 삭제 중 오류 (무시): {cleanup_error}")

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