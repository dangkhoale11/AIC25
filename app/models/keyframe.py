from app.models.factories import keyframe_model_factory

KeyframeBatch1 = keyframe_model_factory("keyframe_batch1")
KeyframeBatch2 = keyframe_model_factory("keyframe_batch2")

# Alias mặc định (cho code cũ không bị lỗi ImportError)
Keyframe = (KeyframeBatch1, KeyframeBatch2)