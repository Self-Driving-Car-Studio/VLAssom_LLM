import uvicorn
import socketio
import asyncio
import sys
import os
import base64
import numpy as np
import cv2
from typing import Dict
from core.router import Router

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Socket.IO 서버 생성
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = socketio.ASGIApp(sio)

# 세션 관리: 접속한 유저(sid)마다 별도의 Router 인스턴스를 가짐
# [주의] Router 클래스가 위에서 정의되어 있어야 타입 힌트 에러가 나지 않습니다.
sessions: Dict[str, Router] = {}

@sio.event
async def connect(sid, environ):
    print(f"✅ 클라이언트 연결됨: {sid}")
    # 연결 시 해당 유저를 위한 라우터 생성
    try:
        sessions[sid] = Router()
    except Exception as e:
        print(f"🚨 Router 생성 실패: {e}")

@sio.event
async def disconnect(sid):
    print(f"❌ 클라이언트 연결 끊김: {sid}")
    if sid in sessions:
        del sessions[sid]

@sio.on('command')
async def handle_command(sid, data):
    """
    앱 -> 서버: 텍스트 메시지 전송 시
    """
    print(f"📩 수신 ({sid}): {data}")
    user_text = data.get('text', '')
    
    # 해당 유저의 라우터 가져오기
    router = sessions.get(sid)
    if not router:
        # 혹시 세션이 없으면 재생성 시도
        try:
            router = Router()
            sessions[sid] = router
        except Exception as e:
            await sio.emit('command-response', {"text": "서버 내부 오류 발생", "type": "simple"}, to=sid)
            return
    
    # Router.handle 실행 (비동기 쓰레드로 실행 권장 - AI 모델 추론 부하 분산)
    try:
        response_data = await asyncio.to_thread(router.handle, user_text)
        await sio.emit('command-response', response_data, to=sid)
        print(f"📤 전송: {response_data}")
    except Exception as e:
        print(f"🚨 처리 중 에러: {e}")
        await sio.emit('command-response', {"text": "오류가 발생했습니다.", "type": "simple"}, to=sid)


@sio.on('action-confirm')
async def handle_action_confirm(sid, data):
    """
    앱 -> 서버: [네] 버튼 클릭 시
    """
    print(f"🔘 버튼 클릭 수신 (YES): {data}")
    
    router = sessions.get(sid)
    if not router:
        return

    # '네'라고 대답한 것으로 처리하여 Decision 로직 수행
    try:
        response_data = await asyncio.to_thread(router.handle, "네")
        await sio.emit('command-response', response_data, to=sid)
        print(f"📤 실행 완료 응답 전송: {response_data}")
    except Exception as e:
        print(f"🚨 실행 중 에러: {e}")
        await sio.emit('command-response', {"text": "실행 중 오류가 발생했습니다.", "type": "simple"}, to=sid)


@sio.on('identify-face')
async def handle_identify_face(sid, base64_image):
    """
    Expo 앱의 'identify-face' 이벤트를 처리하는 메인 핸들러
    """
    print(f"📸 {sid}로부터 이미지 수신 (크기: {len(base64_image)} bytes)")

    try:
        # --- Base64 이미지 디코딩 ---
        if ',' in base64_image:
            header, base64_data = base64_image.split(',', 1)
        else:
            base64_data = base64_image

        img_data = base64.b64decode(base64_data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print(f"⚠️ {sid}: 이미지 디코딩 실패")
            return

        # ... (얼굴 인식 로직 시뮬레이션) ...
        await asyncio.sleep(0.5) 
        user = {"id": "p123", "name": "김블라"}

        # --- 클라이언트로 응답 전송 ---
        await sio.emit('auth-success', user, to=sid)
        print(f"✅ {sid}에게 인증 성공 전송: {user['name']}")

    except Exception as e:
        print(f"🚨 처리 중 오류 발생: {e}")
        await sio.emit('auth-fail', to=sid)

if __name__ == "__main__":
    print("🚀 AI Router 서버 시작 (Port: 3000)")
    uvicorn.run(app, host="0.0.0.0", port=3000)