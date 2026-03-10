import os
from huggingface_hub import HfApi, login

# 1) Đăng nhập (nếu chưa login bằng huggingface-cli)
#    Nếu bạn đã chạy `huggingface-cli login` rồi thì có thể bỏ qua login() ở đây.
HF_TOKEN = "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"  # hoặc để None nếu dùng token đã lưu sẵn
login(token=HF_TOKEN)

# 2) Cấu hình đường dẫn local + repo_id trên Hub
PARENT_FOLDER = r"E:/Project/video_analysis/voice_separator/no_sound_effect"  # folder cha chứa nhiều folder con lớn
REPO_ID = "Coed66/dlsite_asmr"                       # repo private của bạn trên Hub
REPO_TYPE = "dataset"

# 3) (Tuỳ chọn) In ra cấu trúc để chắc chắn đúng folder
print("Uploading from local folder:", PARENT_FOLDER)
for name in os.listdir(PARENT_FOLDER):
    path = os.path.join(PARENT_FOLDER, name)
    if os.path.isdir(path):
        print("[DIR] ", name)
    else:
        print("[FILE]", name)

# 4) Upload toàn bộ nội dung folder cha lên repo (giữ cấu trúc subfolder)
api = HfApi(token=HF_TOKEN)

api.upload_large_folder(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    folder_path=PARENT_FOLDER,   # toàn bộ folder cha, bên trong có các folder con lớn
    num_workers=8,               # tăng/giảm tuỳ CPU/mạng
    # ignore_patterns=["*.tmp", "*.log"],  # nếu muốn bỏ qua một số loại file
    # private=True,  # chỉ cần nếu repo chưa tồn tại; repo bạn đã private rồi thì có/không đều được
)

print("DONE: Uploaded parent folder to:", REPO_ID)
