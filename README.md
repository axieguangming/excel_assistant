# Excel AI 助手

Windows 桌面版 Excel 数据助手。用户上传最大 50 MB 的 `.xlsx`、`.xls` 或 `.csv` 文件后，可以用自然语言请求筛选、排序、分组汇总、Top N、去重、缺失值填充、描述统计和基础图表。数据处理通过经过校验的工具执行，不运行模型生成的 Python 代码。

## Windows 桌面版

桌面客户端由 `desktop.py` 启动本地 FastAPI 服务并在 pywebview 窗口中显示界面。客户端直接通过 HTTPS 调用远程模型 API，不需要单独部署本项目的 API 后端。

**发布密钥风险：** 按当前选定方案，构建时提供的模型 API Key 会写入安装包的 `llm_config.json`。安装包使用者可以提取该密钥并消耗账户额度、产生费用；密钥不能视为保密。发布者应使用专用且限额的 API Key，并配置服务商侧的额度/速率限制。不要把真实密钥提交到 Git、聊天或公开日志中。

### 构建 Windows 安装包

构建机需要 64 位 Python 3.12、项目依赖、`requirements-build.txt` 中的 PyInstaller，以及 Inno Setup 6。通过构建环境变量或项目根目录未提交的 `.env` 提供配置：

```ini
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

然后执行：

```powershell
python -m pip install -r requirements.txt -r requirements-build.txt
.\packaging\build_windows.ps1 -PythonPath "D:\path\to\python.exe"
```

配置生成器要求有效 API Key 和 HTTPS Base URL；不会将密钥打印到控制台。安装包输出到 `release/`。用户电脑需安装 [Microsoft Edge WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/)；Windows 11 通常已包含该运行时。当前安装包尚未配置代码签名证书，公开发布时 Windows SmartScreen 可能显示未知发布者警告。

正式发布前还需配置代码签名、隐私政策和模型服务商披露。用户上传的工作簿及提问会从本机直接发送给配置的模型服务商，请勿上传不应离开本机的敏感数据。桌面客户端本地生成的数据保存在 `%LOCALAPPDATA%\ExcelAssistant`；桌面发行版会禁用训练样本记录。

## 开发运行

远程模型开发配置可放在项目根目录的 `.env` 中（不要提交真实密钥），然后运行：

```powershell
python desktop.py
```

网页调试入口仍可通过 `python app.py` 启动。通过 `LLM_PROVIDER=local` 可在开发环境使用 Ollama 等本地 OpenAI 兼容模型。Excel 加载项相关说明见 [addin/README.md](./addin/README.md)。

## 许可

本软件仅限个人、非商业用途；允许转发版权方发布的未修改官方安装包，不允许商业使用、修改或发布衍生版本。详见 [LICENSE.txt](./LICENSE.txt)。
