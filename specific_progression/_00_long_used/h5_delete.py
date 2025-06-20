import os
import glob

# 삭제할 파일들이 위치한 최상위 폴더 경로
folder_path = '/home/bcml1'

# 폴더 내 및 하위 폴더의 h5 파일 전체 경로 패턴 (재귀적 검색)
pattern = os.path.join(folder_path, '**', '*.keras')

# 패턴에 해당하는 파일 목록을 순회하며 삭제
for file_path in glob.glob(pattern, recursive=True):
    try:
        os.remove(file_path)
        print(f"삭제 완료: {file_path}")
    except Exception as e:
        print(f"파일 삭제 중 오류 발생 ({file_path}): {e}")
