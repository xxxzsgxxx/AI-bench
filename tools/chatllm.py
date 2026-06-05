#!/usr/bin/env python3
"""
增强版大模型对话客户端 v0.0.1-scott
支持: 模型扫描/切换、API Key管理、系统提示词、参数调整、对话导出、连接速度测试、
      自动重试、ITL统计、启动时交互选择模型、Markdown渲染、流式/非流式切换等
"""

import os
import sys
import time
import json
import readline
import requests
from datetime import datetime
from argparse import ArgumentParser
from typing import Dict, List, Optional, Any, Tuple

# 尝试导入 rich 用于 Markdown 渲染
try:
    from rich.console import Console
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

VERSION = "0.0.1-scott"

# 重试配置
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.5
RETRY_STATUS_FORCELIST = {429, 500, 502, 503, 504}


def retry_request(func, *args, **kwargs):
    """带重试的请求包装器"""
    retries = 0
    last_exception = None
    while retries <= MAX_RETRIES:
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            last_exception = e
            retries += 1
            if retries > MAX_RETRIES:
                break
            if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                if e.response.status_code not in RETRY_STATUS_FORCELIST:
                    raise
            wait_time = RETRY_BACKOFF_FACTOR ** retries
            print(f"\033[33m[网络请求失败，{wait_time:.1f}秒后进行第{retries}次重试...]\033[0m")
            time.sleep(wait_time)
    raise last_exception


class ConfigManager:
    """配置管理器"""

    DEFAULT_CONFIG = {
        "api_url": "https://api.openai.com",
        "api_key": "key",
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 2048,
        "system_prompt": None,
        "streaming_mode": False,   # 新增：默认非流式，以支持 Markdown 渲染
    }

    def __init__(self):
        self.config_file = os.path.expanduser("~/.chatllm_config.json")
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        config = self.DEFAULT_CONFIG.copy()
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    saved = json.load(f)
                    config.update(saved)
            except:
                pass
        if "OPENAI_API_KEY" in os.environ:
            config["api_key"] = os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            config["api_url"] = os.environ["OPENAI_BASE_URL"]
        if "OPENAI_API_URL" in os.environ:
            config["api_url"] = os.environ["OPENAI_API_URL"]
        if "OPENAI_MODEL" in os.environ:
            config["model"] = os.environ["OPENAI_MODEL"]
        return config

    def save_config(self):
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            return True
        except:
            return False

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.DEFAULT_CONFIG:
                self.config[key] = value


class SessionManager:
    """多会话管理系统"""

    def __init__(self):
        self.sessions = {}
        self.current_id = 0

    def create_session(self, system_prompt: Optional[str] = None):
        self.current_id += 1
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        self.sessions[self.current_id] = {
            "messages": messages,
            "created": time.time(),
            "token_usage": 0,
            "metrics": {},
        }
        return self.current_id

    def clear_current_session(self):
        if self.current_id in self.sessions:
            session = self.sessions[self.current_id]
            system_msgs = [m for m in session["messages"] if m["role"] == "system"]
            session["messages"] = system_msgs
            session["token_usage"] = 0
            session["metrics"] = {}


class ChatClient:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.api_url = self._normalize_api_url(self.config.config["api_url"])
        self.headers = {"Authorization": f"Bearer {self.config.config['api_key']}"}
        self.model = self.config.config["model"]
        self.sessions = SessionManager()
        self.current_session = self.sessions.create_session(
            self.config.config.get("system_prompt")
        )
        self.history_file = os.path.expanduser("~/.chat_client_history")
        self.available_models = []
        self.streaming_mode = self.config.config.get("streaming_mode", False)
        self.markdown_enabled = RICH_AVAILABLE
        if not self.markdown_enabled:
            print("\033[33m[提示] 未安装 rich 库，Markdown 渲染将使用纯文本。可用 pip install rich 安装以获得更好体验。\033[0m")
        self._init_readline()
        self._init_log_file()

    def _init_log_file(self):
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = f"_chatllm_{today}.log"

    def _append_to_log(self, user_msg: str, assistant_msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] User:\n{user_msg}\n\n")
                f.write(f"[{timestamp}] Assistant:\n{assistant_msg}\n\n")
                f.write("-" * 60 + "\n")
        except Exception as e:
            print(f"\033[31m[日志保存失败] {str(e)}\033[0m")

    def test_connection_speed(self):
        print("\033[33m[正在测试连接速度...]\033[0m")
        try:
            start = time.time()
            response = retry_request(
                requests.get,
                f"{self.api_url}/models",
                headers=self.headers,
                timeout=10
            )
            elapsed = time.time() - start
            if response.status_code == 200:
                print(f"\033[32m✅ 连接成功 | 延迟: {elapsed*1000:.1f} ms\033[0m")
            else:
                print(f"\033[31m❌ 连接失败 | HTTP {response.status_code}\033[0m")
            return elapsed
        except Exception as e:
            print(f"\033[31m❌ 连接失败 | 错误: {str(e)}\033[0m")
            return None

    def update_api_key(self, api_key: str):
        self.config.update(api_key=api_key)
        self.headers = {"Authorization": f"Bearer {api_key}"}
        print(f"\033[33m[API Key 已更新]\033[0m")

    def update_api_url(self, api_url: str):
        self.api_url = self._normalize_api_url(api_url)
        self.config.update(api_url=api_url)
        print(f"\033[33m[API URL 已更新: {self.api_url}]\033[0m")

    def _normalize_api_url(self, url):
        url = url.rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        return url

    def _init_readline(self):
        try:
            import atexit
            readline.parse_and_bind("tab: complete")
            readline.set_completer(self._completer)
            readline.set_history_length(100)
            if os.path.exists(self.history_file):
                try:
                    readline.read_history_file(self.history_file)
                except:
                    pass
            atexit.register(self._save_history)
        except ImportError:
            pass

    def _save_history(self):
        try:
            readline.write_history_file(self.history_file)
        except:
            pass

    def _completer(self, text, state):
        commands = [
            "/new", "/list", "/switch", "/exit", "/models", "/model",
            "/select", "/key", "/url", "/system", "/params", "/export",
            "/config", "/clear", "/save", "/ping", "/help", "/stream", "/markdown"
        ]
        matches = [c for c in commands if c.startswith(text)]
        return matches[state] if state < len(matches) else None

    def _fetch_models(self) -> List[str]:
        try:
            response = retry_request(
                requests.get,
                f"{self.api_url}/models",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data", [])
            return [m["id"] for m in models]
        except Exception as e:
            print(f"\033[31m扫描模型失败: {str(e)}\033[0m")
            return []

    def list_models(self, show_ui=True):
        self.available_models = self._fetch_models()
        if not show_ui:
            return
        if not self.available_models:
            print("\033[31m未找到可用模型\033[0m")
            return
        print(f"\n\033[33m[可用模型列表 ({len(self.available_models)} 个)]\033[0m")
        for i, model_id in enumerate(self.available_models, 1):
            marker = " \033[32m*\033[0m" if model_id == self.model else ""
            print(f"{i:3d}. {model_id}{marker}")
        print("\n\033[36m提示：输入 /model <数字> 或 /select <数字> 快速切换模型\033[0m")

    def prompt_for_model_selection(self):
        print("\033[33m[正在连接API并扫描可用模型...]\033[0m")
        self.available_models = self._fetch_models()
        if not self.available_models:
            print("\033[31m无法获取模型列表，使用配置中的默认模型\033[0m")
            return
        if len(self.available_models) == 1:
            self.model = self.available_models[0]
            self.config.update(model=self.model)
            print(f"\033[33m[唯一可用模型: {self.model}，已自动选择]\033[0m")
            return
        print("\n\033[33m[请选择要使用的模型]\033[0m")
        for i, model_id in enumerate(self.available_models, 1):
            print(f"{i}. {model_id}")
        while True:
            try:
                choice = input(f"\033[36m请输入数字 (1-{len(self.available_models)}): \033[0m").strip()
                if not choice:
                    continue
                idx = int(choice) - 1
                if 0 <= idx < len(self.available_models):
                    self.model = self.available_models[idx]
                    self.config.update(model=self.model)
                    print(f"\033[33m[已选择模型: {self.model}]\033[0m")
                    break
                else:
                    print(f"\033[31m请输入1到{len(self.available_models)}之间的数字\033[0m")
            except ValueError:
                print("\033[31m请输入有效数字\033[0m")

    def switch_model(self, model_identifier: str):
        if model_identifier.isdigit():
            idx = int(model_identifier) - 1
            if 0 <= idx < len(self.available_models):
                model_name = self.available_models[idx]
                self.model = model_name
                self.config.update(model=model_name)
                print(f"\033[33m[已切换至模型: {model_name} (序号 {model_identifier})]\033[0m")
                return True
            else:
                print(f"\033[31m错误: 序号 {model_identifier} 超出范围 (1-{len(self.available_models)})\033[0m")
                return False
        else:
            self.model = model_identifier
            self.config.update(model=model_identifier)
            print(f"\033[33m[已切换至模型: {model_identifier}]\033[0m")
            return True

    def set_system_prompt(self, prompt: str):
        session = self.sessions.sessions[self.current_session]
        session["messages"] = [m for m in session["messages"] if m["role"] != "system"]
        if prompt:
            session["messages"].insert(0, {"role": "system", "content": prompt})
            self.config.update(system_prompt=prompt)
            print(f"\033[33m[系统提示词已设置]\033[0m")
        else:
            self.config.update(system_prompt=None)
            print("\033[33m[系统提示词已清除]\033[0m")

    def update_params(self, **kwargs):
        valid_params = {}
        for key in ["temperature", "top_p", "max_tokens"]:
            if key in kwargs:
                valid_params[key] = kwargs[key]
        self.config.update(**valid_params)
        for key, value in valid_params.items():
            print(f"\033[33m[{key} = {value}]\033[0m")

    def show_config(self):
        print("\n\033[33m[当前配置]")
        print(f"版本: {VERSION}")
        print(f"API URL: {self.api_url}")
        print(f"API Key: {self.config.config['api_key'][:20]}...")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.config.config['temperature']}")
        print(f"Top P: {self.config.config['top_p']}")
        print(f"Max Tokens: {self.config.config['max_tokens']}")
        print(f"流式模式: {'开启' if self.streaming_mode else '关闭'}")
        print(f"Markdown渲染: {'启用' if self.markdown_enabled else '禁用'}")
        system_prompt = self.config.config.get("system_prompt")
        if system_prompt:
            print(f"System Prompt: {system_prompt[:50]}...")
        print(f"当前会话: {self.current_session}")
        print(f"会话总数: {len(self.sessions.sessions)}")
        print(f"日志文件: {self.log_file}\033[0m")

    def export_conversation(self, filename: str):
        session = self.sessions.sessions[self.current_session]
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# 对话导出 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for msg in session["messages"]:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        f.write(f"**System:** {content}\n\n")
                    elif role == "user":
                        f.write(f"**User:** {content}\n\n")
                    elif role == "assistant":
                        f.write(f"**Assistant:** {content}\n\n")
            print(f"\033[33m[对话已导出至: {filename}]\033[0m")
        except Exception as e:
            print(f"\033[31m导出失败: {str(e)}\033[0m")

    def _handle_command(self, input_cmd):
        cmd = input_cmd.strip()
        cmd_lower = cmd.lower()

        if cmd_lower == "/new":
            new_id = self.sessions.create_session(self.config.config.get("system_prompt"))
            print(f"\033[33m[新会话 {new_id} 已创建]\033[0m")
            return True

        elif cmd_lower.startswith("/switch"):
            try:
                parts = cmd.split()
                if len(parts) < 2:
                    print("\033[31m用法: /switch <会话ID>\033[0m")
                    return True
                session_id = int(parts[1])
                if session_id in self.sessions.sessions:
                    self.current_session = session_id
                    print(f"\033[33m[已切换至会话 {session_id}]\033[0m")
                else:
                    print("\033[31m错误: 会话ID不存在\033[0m")
            except ValueError:
                print("\033[31m用法: /switch <会话ID>\033[0m")
            return True

        elif cmd_lower == "/list":
            print("\n\033[33m[活跃会话列表]")
            for sid, sess in self.sessions.sessions.items():
                stats = f"消息数: {len(sess['messages']) // 2} | Tokens: {sess['token_usage']}"
                marker = " *" if sid == self.current_session else ""
                print(f"{sid}.\t{stats}{marker}")
            return True

        elif cmd_lower == "/models":
            self.list_models(show_ui=True)
            return True

        elif cmd_lower.startswith("/model "):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print("\033[31m用法: /model <模型名称或序号>\033[0m")
            else:
                self.switch_model(parts[1])
            return True

        elif cmd_lower.startswith("/select "):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print("\033[31m用法: /select <数字序号>\033[0m")
            else:
                self.switch_model(parts[1])
            return True

        elif cmd_lower.startswith("/key "):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print("\033[31m用法: /key <API_KEY>\033[0m")
            else:
                self.update_api_key(parts[1])
            return True

        elif cmd_lower.startswith("/url "):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print("\033[31m用法: /url <API_URL>\033[0m")
            else:
                self.update_api_url(parts[1])
            return True

        elif cmd_lower.startswith("/system"):
            parts = cmd.split(None, 1)
            prompt = parts[1] if len(parts) > 1 else ""
            self.set_system_prompt(prompt)
            return True

        elif cmd_lower.startswith("/params"):
            parts = cmd.split()[1:]
            if not parts:
                print("\033[33m用法: /params temp=<value> top_p=<value> max_tokens=<value>\033[0m")
                print("\033[33m示例: /params temp=0.8 top_p=0.9\033[0m")
                return True
            params = {}
            for part in parts:
                if "=" in part:
                    key, value = part.split("=", 1)
                    try:
                        if key in ["temp", "temperature"]:
                            params["temperature"] = float(value)
                        elif key == "top_p":
                            params["top_p"] = float(value)
                        elif key in ["max_tokens", "tokens"]:
                            params["max_tokens"] = int(value)
                    except ValueError:
                        print(f"\033[31m无效参数: {part}\033[0m")
            if params:
                self.update_params(**params)
            return True

        elif cmd_lower.startswith("/export"):
            parts = cmd.split(None, 1)
            if len(parts) < 2:
                print("\033[31m用法: /export <文件名>\033[0m")
            else:
                self.export_conversation(parts[1])
            return True

        elif cmd_lower == "/config":
            self.show_config()
            return True

        elif cmd_lower == "/clear":
            self.sessions.clear_current_session()
            print("\033[33m[当前会话已清空]\033[0m")
            return True

        elif cmd_lower == "/save":
            if self.config.save_config():
                print("\033[33m[配置已保存至 ~/.chatllm_config.json]\033[0m")
            else:
                print("\033[31m配置保存失败\033[0m")
            return True

        elif cmd_lower == "/ping":
            self.test_connection_speed()
            return True

        elif cmd_lower == "/stream":
            self.streaming_mode = not self.streaming_mode
            self.config.update(streaming_mode=self.streaming_mode)
            print(f"\033[33m[流式模式已{'开启' if self.streaming_mode else '关闭'}]\033[0m")
            return True

        elif cmd_lower == "/markdown":
            if not RICH_AVAILABLE:
                print("\033[31m[错误] 未安装 rich 库，无法启用 Markdown 渲染。请运行 pip install rich\033[0m")
                return True
            self.markdown_enabled = not self.markdown_enabled
            print(f"\033[33m[Markdown 渲染已{'启用' if self.markdown_enabled else '禁用'}]\033[0m")
            return True

        elif cmd_lower == "/help":
            self._show_help()
            return True

        return False

    def _show_help(self):
        help_text = f"""
\033[33m[命令帮助] 版本 {VERSION}

会话管理:
  /new              创建新会话
  /list             列出所有会话
  /switch <id>      切换到指定会话
  /clear            清空当前会话历史

模型管理:
  /models           扫描并列出可用模型（支持数字序号）
  /model <名称|序号> 切换模型，例如 /model 3 或 /model gpt-4
  /select <序号>    同 /model <序号>

API配置:
  /key <api_key>    设置API Key
  /url <base_url>   设置API Base URL
  /config           显示当前配置
  /save             保存配置到文件
  /ping             测试API连接速度

对话增强:
  /system <prompt>  设置系统提示词
  /params ...       设置生成参数
  /export <file>    导出当前对话
  /stream           切换流式/非流式模式（非流式支持Markdown渲染）
  /markdown         切换Markdown渲染开关（需rich库）

其他:
  /help             显示此帮助
  exit/quit         退出程序

注意: 对话历史会自动保存到 _chatllm_YYYY-MM-DD.log 文件中
\033[0m"""
        print(help_text)

    def _show_metrics(self, metrics):
        if not metrics["first_token"]:
            print("\033[31m未收到有效响应\033[0m")
            return
        total_time = time.time() - metrics["start_time"]
        generation_time = metrics["last_token"] - metrics["first_token"]
        avg_itl_ms = metrics.get("avg_itl_ms", 0.0)
        print(f"\n\033[33m[性能统计]")
        print(f"首Token延迟: {metrics['first_token'] - metrics['start_time']:.2f}s")
        print(f"总生成时间: {generation_time:.2f}s")
        print(f"总Token数量: {metrics['total_tokens']}")
        print(f"生成速度: {metrics['total_tokens'] / generation_time:.1f} tokens/s")
        print(f"端到端速度: {metrics['total_tokens'] / total_time:.1f} tokens/s")
        print(f"平均ITL: {avg_itl_ms:.1f} ms/token\033[0m")

    def _render_markdown(self, text: str):
        """使用 rich 渲染 Markdown 并打印"""
        if self.markdown_enabled and RICH_AVAILABLE:
            console = Console()
            md = Markdown(text)
            console.print(md)
        else:
            # 纯文本输出
            print(text)

    def chat_non_streaming(self, prompt: str):
        """非流式对话：收集完整回复，支持 Markdown 渲染"""
        session = self.sessions.sessions[self.current_session]
        session["messages"].append({"role": "user", "content": prompt})

        metrics = {
            "start_time": time.time(),
            "first_token": None,
            "last_token": None,
            "total_tokens": 0,
            "avg_itl_ms": 0.0,
        }

        request_body = {
            "model": self.model,
            "messages": session["messages"],
            "stream": False,  # 非流式
            "temperature": self.config.config["temperature"],
            "top_p": self.config.config["top_p"],
            "max_tokens": self.config.config["max_tokens"],
        }

        retries = 0
        last_exception = None
        response = None
        while retries <= MAX_RETRIES:
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=request_body,
                    timeout=60,
                )
                response.raise_for_status()
                break
            except Exception as e:
                last_exception = e
                retries += 1
                if retries > MAX_RETRIES:
                    break
                wait_time = RETRY_BACKOFF_FACTOR ** retries
                print(f"\033[33m[请求失败，{wait_time:.1f}秒后进行第{retries}次重试...]\033[0m")
                time.sleep(wait_time)

        if response is None:
            print(f"\n\033[31m请求失败（重试{MAX_RETRIES}次后）: {last_exception}\033[0m")
            session["messages"].pop()
            return

        try:
            data = response.json()
            assistant_reply = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)

            # 更新指标（非流式无法精确计算首token延迟等，做近似）
            now = time.time()
            metrics["first_token"] = now
            metrics["last_token"] = now
            metrics["total_tokens"] = total_tokens
            session["token_usage"] += total_tokens
            session["metrics"] = metrics

            # 输出助手回复（支持 Markdown 渲染）
            print("\033[32mAssistant:\033[0m ")
            self._render_markdown(assistant_reply)

            # 保存对话
            session["messages"].append({"role": "assistant", "content": assistant_reply})
            self._append_to_log(prompt, assistant_reply)

            # 性能统计（非流式缺少细粒度时间，仅显示基本信息）
            print(f"\n\033[33m[性能统计]")
            print(f"总耗时: {now - metrics['start_time']:.2f}s")
            print(f"总Token数量: {total_tokens}")
            print(f"生成速度: {total_tokens / (now - metrics['start_time']):.1f} tokens/s\033[0m")

        except Exception as e:
            print(f"\n\033[31m解析响应错误: {str(e)}\033[0m")
            session["messages"].pop()

    def chat_streaming(self, prompt: str):
        """流式对话（原始模式，不支持 Markdown 实时渲染）"""
        session = self.sessions.sessions[self.current_session]
        session["messages"].append({"role": "user", "content": prompt})

        metrics = {
            "start_time": time.time(),
            "first_token": None,
            "last_token": None,
            "total_tokens": 0,
            "avg_itl_ms": 0.0,
        }

        request_body = {
            "model": self.model,
            "messages": session["messages"],
            "stream": True,
            "temperature": self.config.config["temperature"],
            "top_p": self.config.config["top_p"],
            "max_tokens": self.config.config["max_tokens"],
        }

        retries = 0
        last_exception = None
        response = None
        while retries <= MAX_RETRIES:
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=request_body,
                    stream=True,
                    timeout=30,
                )
                response.raise_for_status()
                break
            except Exception as e:
                last_exception = e
                retries += 1
                if retries > MAX_RETRIES:
                    break
                wait_time = RETRY_BACKOFF_FACTOR ** retries
                print(f"\033[33m[请求失败，{wait_time:.1f}秒后进行第{retries}次重试...]\033[0m")
                time.sleep(wait_time)

        if response is None:
            print(f"\n\033[31m请求失败（重试{MAX_RETRIES}次后）: {last_exception}\033[0m")
            session["messages"].pop()
            return

        try:
            print("\033[32mAssistant:\033[0m ", end="", flush=True)
            full_response = []
            prev_time = None
            interval_sum = 0.0
            interval_count = 0

            for chunk in response.iter_lines():
                if chunk:
                    data = chunk.decode("utf-8").lstrip("data: ").strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        json_data = json.loads(data)
                        content = (
                            json_data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                    except json.JSONDecodeError:
                        continue

                    if content:
                        now = time.time()
                        metrics["total_tokens"] += len(content)
                        metrics["last_token"] = now
                        if not metrics["first_token"]:
                            metrics["first_token"] = now

                        if prev_time is not None:
                            interval = now - prev_time
                            interval_sum += interval
                            interval_count += 1
                        prev_time = now

                        print(content, end="", flush=True)
                        full_response.append(content)

            if interval_count > 0:
                metrics["avg_itl_ms"] = (interval_sum / interval_count) * 1000.0

            assistant_reply = "".join(full_response)
            session["messages"].append({"role": "assistant", "content": assistant_reply})
            session["token_usage"] += metrics["total_tokens"]
            session["metrics"] = metrics

            self._append_to_log(prompt, assistant_reply)
            self._show_metrics(metrics)

        except Exception as e:
            print(f"\n\033[31m流式处理错误: {str(e)}\033[0m")
            session["messages"].pop()
        finally:
            response.close()

    def send_message(self, prompt: str):
        """根据当前模式选择发送方式"""
        if self.streaming_mode:
            self.chat_streaming(prompt)
        else:
            self.chat_non_streaming(prompt)


def interactive_setup():
    print("\033[33m=== 大模型对话客户端配置 ===\033[0m\n")
    api_url = input("API Base URL [https://api.openai.com]: ").strip()
    if not api_url:
        api_url = "https://api.openai.com"
    api_key = input("API Key [从环境变量/配置文件读取]: ").strip()
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            config_file = os.path.expanduser("~/.chatllm_config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                        api_key = config.get("api_key", "key")
                except:
                    api_key = "key"
            else:
                api_key = "key"
    model = input("模型名称 [gpt-3.5-turbo]: ").strip()
    if not model:
        model = "gpt-3.5-turbo"
    return api_url, api_key, model


if __name__ == "__main__":
    parser = ArgumentParser(description="增强版大模型对话客户端")
    parser.add_argument("--base-url", help="API服务地址")
    parser.add_argument("--api-key", help="API认证密钥")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("-i", "--interactive", action="store_true", help="交互式配置")
    args = parser.parse_args()

    config = ConfigManager()

    if args.base_url:
        config.update(api_url=args.base_url)
    if args.api_key:
        config.update(api_key=args.api_key)
    if args.model:
        config.update(model=args.model)

    if args.interactive:
        api_url, api_key, model = interactive_setup()
        config.update(api_url=api_url, api_key=api_key, model=model)

    client = ChatClient(config)

    if not args.model:
        client.prompt_for_model_selection()

    print(f"\n\033[33m{'=' * 50}")
    print(f"增强版大模型对话客户端 v{VERSION}")
    print(f"{'=' * 50}\033[0m")
    print(f"\n当前配置:")
    print(f"  API URL: {client.api_url}")
    print(f"  Model: {client.model}")
    print(f"  流式模式: {'开启' if client.streaming_mode else '关闭'} (输入 /stream 切换)")
    if not client.streaming_mode and client.markdown_enabled:
        print(f"  Markdown渲染: 已启用 (输入 /markdown 切换)")
    print(f"\n输入 \033[36m/help\033[0m 查看帮助, \033[36mexit\033[0m 退出\n")

    try:
        while True:
            try:
                user_input = input("\033[34mYou:\033[0m ")
            except KeyboardInterrupt:
                print("\n\033[33m对话已终止\033[0m")
                break
            except EOFError:
                print("\n\033[33m再见!\033[0m")
                break

            if user_input.lower() in ("exit", "quit"):
                break

            if not user_input.strip():
                continue

            if client._handle_command(user_input):
                continue

            client.send_message(user_input)

    finally:
        pass
