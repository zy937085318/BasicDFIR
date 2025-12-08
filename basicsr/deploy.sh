#!/bin/bash
set -e
cd /var/www/myrepo
git fetch origin
git reset --hard origin/main   # 强制同步到最新
npm ci --production            # 按你项目需要
pm2 reload app                 # 或 systemctl restart xxx
echo "deploy done"