import uvicorn
import socketio
import asyncio
import sys
import os
import base64
import numpy as np
import cv2
from typing import Dict, Any, Union, Optional
import base64
import uuid
import asyncio

# 커스텀 모듈
from core.router import Router
from core.model_loader import ModelContainer

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
    앱 -> 서버: 음성 데이터 수신 -> STT -> Router -> 응답
    """
    print(f"🎤 오디오 데이터 수신 ({sid})")
    
    router = get_or_create_router(sid)
    if not router:
        return

    try:
        # 1. 데이터 파싱 및 저장 준비
        b64_string = data.get('audioData')
        file_ext = data.get('format', 'm4a')
        user_id = data.get('userId', 'unknown')

        # Base64 디코딩
        audio_bytes = base64.b64decode(b64_string)
        
        # uploads 폴더 확보
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
            
        # 임시 파일명 생성 및 저장
        filename = f"uploads/{user_id}_{uuid.uuid4()}.{file_ext}"
        
        # [주의] 파일 쓰기는 동기 작업이므로, 비동기 래핑 없이 쓸 땐 빠를수록 좋음
        # 파일이 크다면 aiofiles 라이브러리 고려, 지금은 그냥 진행
        with open(filename, "wb") as f:
            f.write(audio_bytes)
            
        print(f"💾 파일 저장 완료: {filename}")

        # -------------------------------------------------------
        # 2. Whisper STT 변환 (오래 걸리므로 별도 스레드 실행)
        # -------------------------------------------------------
        print("👂 음성 인식 중...")
        
        # 모델 가져오기
        stt_model = global_models.stt_model
        
        # 실제 추론 실행 함수 (내부 함수로 정의하거나 별도로 뺌)
        def transcribe_audio():
            # fp16=False는 CPU 경고 방지용 (GPU 있으면 제거 가능)
            return stt_model.transcribe(filename, language="ko", fp16=False)

        # 쓰레드에서 실행 (서버 멈춤 방지)
        result = await asyncio.to_thread(transcribe_audio)
        recognized_text = result['text'].strip()
        
        print(f"🗣️ 인식된 텍스트: \"{recognized_text}\"")

        # -------------------------------------------------------
        # 3. 인식된 텍스트를 Router(두뇌)에 전달
        # -------------------------------------------------------
        if not recognized_text:
            await sio.emit('command-response', {"text": "음성이 잘 들리지 않았어요.", "type": "simple"}, to=sid)
            return
        
        await sio.emit('user-speech', {'text': recognized_text}, to=sid)

        # [핵심] 텍스트가 된 명령어를 기존 handle 함수에 그대로 넣습니다!
        response_data = await asyncio.to_thread(router.handle, recognized_text)
        
        # 4. 결과 응답 (format_response_payload 헬퍼 사용)
        payload = format_response_payload(response_data)


        await sio.emit('command-response', payload, to=sid)
        print(f"📤 응답 전송: {payload}")

        # 5. (선택) 임시 파일 삭제 (용량 관리)
        os.remove(filename)

    except Exception as e:
        print(f"🚨 오디오 처리 중 에러: {e}")
        await sio.emit('command-response', {"text": "음성 처리에 실패했습니다.", "type": "error"}, to=sid)

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