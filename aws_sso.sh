#!/bin/bash

# 1. iam identity centerに設定しておく
# 今回はIdentity Center ディレクトリにユーザー作って、許可セットとアカくっつけた

# 2. aws-cli v2をインストール
if ! [ -f ../awscliv2.zip ]; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "../awscliv2.zip"
    unzip ../awscliv2.zip -d ..
    sudo ../aws/install
fi

# 3. ~/.aws/configにsso用のやつを書いておく
touch ~/.aws/config
# cat << EOF > ~/.aws/config
# [default]
# sso_session = codespace_sso_session
# sso_account_id = ${AWS_ACCOUNT_ID}
# sso_role_name = AdministratorAccess
# region = ap-northeast-1
# output = json
# [sso-session codespace_sso_session]
# sso_start_url = ${AWS_SSO_START_URL}
# sso_region = ap-northeast-1
# sso_registration_scopes = sso:account:access
# EOF

cat << EOF > ~/.aws/config
[default]
sso_session = codespace_sso_session
sso_account_id = ${AWS_ACCOUNT_ID_SANDBOX}
sso_role_name = AWSAdministratorAccess
region = ap-northeast-1
output = json
[sso-session codespace_sso_session]
sso_start_url = ${AWS_SSO_START_URL}
sso_region = ap-northeast-1
sso_registration_scopes = sso:account:access
EOF


# 4. ssoログインする
aws sso login

aws sts get-caller-identity