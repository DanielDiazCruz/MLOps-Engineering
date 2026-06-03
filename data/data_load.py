import os, requests

_data_root = './data'
_data_filepath = os.path.join(_data_root, 'Diabetes.csv')

os.makedirs(_data_root, exist_ok=True)

if not os.path.isfile(_data_filepath):
    url = 'https://docs.google.com/uc?export=download&confirm=t&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
    r = requests.get(url, allow_redirects=True, stream=True)
    open(_data_filepath, 'wb').write(r.content)
    print(f'Descargado: {os.path.getsize(_data_filepath)} bytes')
else:
    print('Ya existe el archivo')
