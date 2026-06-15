from cryptography.fernet import Fernet
# copia y pega este valor en secrets.toml
print(Fernet.generate_key().decode())  


""" 
def encrypt_file(src_path: str, dest_path: str):
    with open(src_path, 'rb') as f:
        data = f.read()
    token = fernet.encrypt(data)
    with open(dest_path, 'wb') as f:
        f.write(token)
    if os.path.exists(src_path): os.remove(src_path) """