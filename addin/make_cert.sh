#!/usr/bin/env bash
# 生成供 Excel 加载项本地开发使用的自签名 HTTPS 证书（localhost）
# 依赖 Git Bash 自带的 openssl。
set -e
cd "$(dirname "$0")"
mkdir -p certs

# 写一份最小 openssl 配置（部分 Git for Windows 的 openssl 默认配置文件缺失）
cat > certs/openssl.cnf <<'EOF'
[req]
distinguished_name = req_dn
x509_extensions = v3_ca
prompt = no

[req_dn]
CN = localhost

[v3_ca]
subjectAltName = DNS:localhost, IP:127.0.0.1
basicConstraints = critical, CA:TRUE
keyUsage = critical, digitalSignature, keyEncipherment, keyCertSign
extendedKeyUsage = serverAuth
EOF

openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout certs/key.pem -out certs/cert.pem \
  -days 825 -config certs/openssl.cnf -extensions v3_ca

CERT_WIN="$(cd certs && cygpath -w "$PWD/cert.pem")"

echo
echo "✅ 证书已生成：addin/certs/cert.pem 与 addin/certs/key.pem"
echo
echo "下一步：用【管理员 PowerShell】信任该证书（否则 Excel 打不开）："
echo "  certutil -addstore -f ROOT \"$CERT_WIN\""
echo
echo "然后启动服务（项目根目录、conda 环境 excel_assistant）："
echo "  export SSL_CERTFILE=addin/certs/cert.pem"
echo "  export SSL_KEYFILE=addin/certs/key.pem"
echo "  python app.py    # 将以 https://localhost:8443 提供服务"
