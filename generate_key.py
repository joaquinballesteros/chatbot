from cryptography.fernet import Fernet
# copia y pega este valor en secrets.toml
print(Fernet.generate_key().decode())  
