# Crear el directorio .ssh en el home efímero
mkdir -p ~/.ssh

# Enlazar la llave privada y pública desde /storage
ln -sf /storage/.ssh/id_ed25519 ~/.ssh/id_ed25519
ln -sf /storage/.ssh/id_ed25519.pub ~/.ssh/id_ed25519.pub

# Ajustar permisos (SSH es muy estricto con esto)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519