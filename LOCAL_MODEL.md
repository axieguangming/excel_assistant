# 远程模型与桌面客户端

## 桌面发行版配置

桌面客户端在本机处理工作簿，并直接通过 HTTPS 调用 OpenAI 兼容的远程模型 API；不需要部署本项目的远程后端。上传的文件和提问会发送到模型服务商，发布者应事先披露服务商、数据处理方式和隐私政策。

当前选定的发行方式会把模型 API Key 写入安装包中的 `llm_config.json`。该文件和安装程序都不能保护密钥，用户可以提取并滥用它。只使用专用于该应用、设置了严格额度和速率限制的 Key。不要提交真实 Key、将其粘贴到聊天中，或打印到日志。

## 配置与构建

可在构建机设置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 和可选的 `DEEPSEEK_MODEL` 环境变量，或将它们放在项目根目录未提交的 `.env` 文件中。发布构建需要 API Key 和 HTTPS 地址：

```ini
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

```powershell
python -m pip install -r requirements.txt -r requirements-build.txt
.\packaging\build_windows.ps1 -PythonPath "D:\path\to\python.exe"
```

`packaging/write_model_config.py` 在打包过程中验证配置并生成 `dist\ExcelAssistant\llm_config.json`。密钥不应被加入源码控制；`.gitignore` 忽略生成的模型配置文件。构建完成后，`release/` 中的安装包包含相同的密钥。

构建需要 64 位 Python 3.12、项目依赖、PyInstaller 和 Inno Setup 6。安装包为 64 位 Windows 用户级安装程序，默认不需要管理员权限。目标电脑须有 Microsoft Edge WebView2 Runtime；Windows 11 通常已预装。公开发布还需代码签名证书；没有签名时 Windows SmartScreen 可能显示未知发布者警告。

## 本地模型（开发/实验）

开发时可以在 `.env` 中设置 `LLM_PROVIDER=local` 连接 Ollama 或其他 OpenAI 兼容服务。本地模型及权重不会打包进安装程序。

```ini
LLM_PROVIDER=local
LOCAL_MODEL=qwen2.5-coder:1.5b
LOCAL_BASE_URL=http://localhost:11434/v1
```

本地开发使用远程模型时，`LLM_PROVIDER=remote`（默认）并配置 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`。旧式 `deepseek_api` JSON 配置仍可用于开发兼容，但新发行包使用上文所述的专用 `llm_config.json`。

## 数据与许可

- 上传的文件和分析结果保存在 `%LOCALAPPDATA%\ExcelAssistant`。
- 桌面发行版会禁用训练样本日志。开发服务器日志默认关闭；开启 `TRAIN_DATA_LOGGING=1` 可能记录提示词和表格样例。
- 软件仅限个人、非商业使用。允许转发官方发布的未修改安装包，不允许修改或发布衍生版本，详见 [LICENSE.txt](./LICENSE.txt)。
