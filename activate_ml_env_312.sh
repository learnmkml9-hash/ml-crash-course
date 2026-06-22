source .venv312/bin/activate
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$(python -c "import certifi; print(certifi.where())")
