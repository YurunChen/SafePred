# Tiktoken 下载失败问题解决方案

## 为什么下载会失败？

### 1. 网络连接问题
- **SSL/TLS 握手失败**：在建立 HTTPS 连接时，SSL 握手阶段连接被重置
- **防火墙/代理阻断**：网络防火墙或代理服务器阻止了对 `openaipublic.blob.core.windows.net` 的访问
- **网络不稳定**：网络连接不稳定导致下载中断

### 2. 目标服务器问题
- **服务器临时不可用**：Azure Blob Storage 可能临时不可用
- **地理位置限制**：某些地区可能无法直接访问 Azure 服务
- **DNS 解析问题**：无法正确解析域名

### 3. 下载 URL
tiktoken 从以下地址下载文件：
- `o200k_base`: `https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken`
- `cl100k_base`: `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`
- 其他编码文件也在同一域名下

## 解决方案

### 方案 1: 使用手动下载脚本（推荐）

使用提供的脚本，支持多种下载方法和重试机制：

```bash
cd /data/chenyurun/project/wasp/visualwebarena
source venv/bin/activate

# 使用 requests 方法（默认）
python scripts/download_tiktoken_manually.py o200k_base

# 使用 curl 方法
python scripts/download_tiktoken_manually.py o200k_base --method curl

# 使用 wget 方法
python scripts/download_tiktoken_manually.py o200k_base --method wget

# 使用代理
python scripts/download_tiktoken_manually.py o200k_base --proxy http://proxy.example.com:8080

# 下载所有文件
python scripts/download_tiktoken_manually.py all
```

### 方案 2: 使用 curl/wget 手动下载

如果 Python 脚本也失败，可以直接使用系统工具：

```bash
# 创建缓存目录
CACHE_DIR=~/.cache/tiktoken
mkdir -p "$CACHE_DIR"

# 计算缓存文件名（URL 的 SHA1）
URL="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
CACHE_KEY=$(echo -n "$URL" | sha1sum | cut -d' ' -f1)
CACHE_PATH="$CACHE_DIR/$CACHE_KEY"

# 使用 curl 下载
curl -L -o "$CACHE_PATH" "$URL"

# 或使用 wget
wget -O "$CACHE_PATH" "$URL"
```

### 方案 3: 配置代理

如果网络需要代理，可以：

```bash
# 设置代理环境变量
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 然后运行下载脚本
python scripts/download_tiktoken_manually.py o200k_base
```

### 方案 4: 从其他机器下载后传输

1. 在有网络的机器上下载文件
2. 传输到目标机器的缓存目录：

```bash
# 在源机器上
python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o-mini')"
# 这会触发下载，文件会缓存在 ~/.cache/tiktoken/

# 复制缓存目录到目标机器
scp -r ~/.cache/tiktoken user@target-machine:~/.cache/
```

### 方案 5: 使用镜像或 CDN

如果 Azure Blob Storage 不可访问，可以尝试：
1. 使用 VPN 或代理
2. 从 GitHub 镜像下载（如果有）
3. 使用其他 CDN 服务

### 方案 6: 修改代码使用本地文件

如果下载持续失败，可以修改 tiktoken 加载逻辑，使用本地文件：

```python
# 在 tokenizers.py 中，可以添加本地文件路径支持
import os
if os.path.exists("/path/to/local/o200k_base.tiktoken"):
    # 使用本地文件
    pass
```

## 验证下载

下载完成后，验证文件是否正确：

```bash
python -c "import tiktoken; enc = tiktoken.encoding_for_model('gpt-4o-mini'); print('Success!')"
```

## 缓存位置

tiktoken 的缓存目录位置（按优先级）：
1. `$TIKTOKEN_CACHE_DIR` 环境变量指定的目录
2. `$DATA_GYM_CACHE_DIR` 环境变量指定的目录
3. `$TMPDIR/data-gym-cache` 或 `/tmp/data-gym-cache`

检查缓存：
```bash
ls -la ~/.cache/tiktoken/  # 或 /tmp/data-gym-cache/
```

## 常见问题

### Q: 为什么需要下载这些文件？
A: tiktoken 需要 BPE（Byte Pair Encoding）文件来正确计算 token 数量。不同模型使用不同的编码。

### Q: 下载的文件有多大？
A: 通常几 MB 到几十 MB，具体取决于编码类型。

### Q: 下载一次后还需要再下载吗？
A: 不需要。文件会缓存在本地，除非缓存被清除或文件损坏。

### Q: 可以离线使用吗？
A: 可以。一旦文件下载并缓存，就可以离线使用。

## 联系支持

如果以上方法都失败，可能需要：
1. 检查网络连接和防火墙设置
2. 联系网络管理员配置代理
3. 考虑使用 VPN 或其他网络解决方案



