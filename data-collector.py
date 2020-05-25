# Must install 'xlrd' for pandas: ```pip install xlrd```
import os
import pandas as pd
import re
import requests

df = pd.read_excel('response.xlsx')
df['id'] = df['Thư điện tử'].str[:8] ## Cột id: Lấy mã SV làm id
df['folder'] = df.agg('{0[id]}_{0[Họ và đệm]} {0[Tên]}'.format, axis=1) # Cột Folder: theo cú pháp MSV_Họ và tên
df = df.sort_values(by=['Được hoàn thành'], ascending = False) # Sort bảng theo Thời gian thực hiện = "Được hoàn thành" - "Bắt đầu vào lúc"

def extract_id(response):
    if type(response) != str:
        return None
    pos = response.find('open?id=')
    if pos == -1:
        pos = response.find('/file/d/')
    if pos != -1:
        pos += 8
        response = response[pos:]
        result = re.search(r"[\w-]+", response)
        return result.group(0)
    else:
        return None

def download_file_from_google_drive(id, destination, replace=False):
    if not replace and os.path.isfile(destination):
        print("Destination file", destination, "exists, download aborted")
        return
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_response(row):
    folder = row['folder']
    response = row['Response 1']    
    data_file_id = extract_id(response) # Lấy id của link google drive
    if data_file_id is not None:
        destination_dir = os.path.join('assignment1', folder) # Tạo path folder
        destination = os.path.join(destination_dir, 'data.zip')
        os.makedirs(destination_dir, exist_ok=True) # Tạo folder
        print("Downloading", data_file_id, '...')
        download_file_from_google_drive(data_file_id, destination)
        print("Done.")
    else:
        print("Cannot find drive file ID, check response")

def download_by_student_id(student_id):
    for r in df.iterrows():
        r = r[1] # r gồm ({id}, {data}) - kiểu dữ liệu: Tuple
        if r['id'] == student_id:
            print(r['folder'], r['Được hoàn thành'], r['Response 1'])
            download_response(r)
            print('---------------')

download_by_student_id('17021288')